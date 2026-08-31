"""AutoOutreach -- personalised cold outreach emails, free to run end to end.

Modes
  --dry-run   (default) generate every email and save it for review; sends nothing
  --send      generate and actually send, respecting delay + daily cap
  --test      generate one email and send it to your own address only

Examples
  python main.py --dry-run
  python main.py --dry-run --limit 3
  python main.py --test
  python main.py --send --limit 10
  python main.py --send --csv other_contacts.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import config
import llm
import mailer
import tracker
from config import ConfigError, Profile, Settings
from tracker import Contact, SentLog

LOG = logging.getLogger("autooutreach")


# ---------------------------------------------------------------------------
# Logging: console + timestamped file, every attempt recorded.
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"autooutreach_{datetime.now():%Y%m%d_%H%M%S}.log"

    root = logging.getLogger("autooutreach")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s %(name)s  %(message)s")
    )
    root.addHandler(file_handler)

    # Windows consoles default to cp1252 and would raise UnicodeEncodeError on
    # characters an LLM may emit (curly quotes, em dashes). Force UTF-8 output.
    stream = sys.stdout
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    console = logging.StreamHandler(stream)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autooutreach",
        description=(
            "Send short, LLM-personalised cold outreach emails "
            "(Groq/Gemini free tiers + Gmail SMTP)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and save emails for review without sending (default).",
    )
    mode.add_argument("--send", action="store_true", help="Actually send the emails.")
    mode.add_argument(
        "--test",
        action="store_true",
        help="Generate one email and send it to your own address as a sanity check.",
    )

    parser.add_argument("--csv", dest="csv_path", help="Path to the contacts CSV.")
    parser.add_argument("--limit", type=int, help="Process at most N contacts this run.")
    parser.add_argument(
        "--delay-min", type=float, help="Override minimum delay between sends (s)."
    )
    parser.add_argument(
        "--delay-max", type=float, help="Override maximum delay between sends (s)."
    )
    parser.add_argument("--daily-cap", type=int, help="Override the per-day send cap.")
    parser.add_argument(
        "--resend",
        action="store_true",
        help="Do not skip contacts already marked SENT in sent_log.csv.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt before a real send.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the models your API keys can actually use, then exit. "
        "Use this when a run fails with model_not_found.",
    )

    args = parser.parse_args(argv)
    if not (args.send or args.test):
        args.dry_run = True  # dry run is the default, always
    return args


def apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.csv_path:
        p = Path(args.csv_path)
        settings.contacts_csv = p if p.is_absolute() else (config.ROOT / p)
    if args.delay_min is not None:
        settings.send_delay_min = args.delay_min
    if args.delay_max is not None:
        settings.send_delay_max = args.delay_max
    if settings.send_delay_max < settings.send_delay_min:
        settings.send_delay_min, settings.send_delay_max = (
            settings.send_delay_max,
            settings.send_delay_min,
        )
    if args.daily_cap is not None:
        settings.daily_send_cap = args.daily_cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "contact"


def select_contacts(
    contacts: List[Contact],
    sent_log: SentLog,
    args: argparse.Namespace,
    log: logging.Logger,
) -> List[Contact]:
    """Filter out malformed addresses, duplicates and already-sent contacts."""
    already = set() if args.resend else sent_log.already_sent()
    seen: Set[str] = set()
    selected: List[Contact] = []

    for c in contacts:
        key = c.email.strip().lower()
        if not mailer.is_valid_email(c.email):
            log.warning(
                "SKIP row %d (%s @ %s): malformed email %r",
                c.row_number,
                c.first_name,
                c.company,
                c.email,
            )
            continue
        if key in already:
            log.info(
                "SKIP %s @ %s: already sent in a previous run.", c.first_name, c.company
            )
            continue
        if key in seen:
            log.warning(
                "SKIP row %d: duplicate address %s in the CSV.", c.row_number, c.email
            )
            continue
        seen.add(key)
        selected.append(c)

    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
    return selected


def list_models(settings: Settings) -> int:
    """Print the models each configured key can reach, marking the one in .env."""
    import requests

    checks = (
        ("Groq", settings.groq_api_key, settings.groq_model, llm.list_groq_models),
        ("Gemini", settings.gemini_api_key, settings.gemini_model, llm.list_gemini_models),
    )
    for label, key, configured, lister in checks:
        print(f"\n=== {label} ===")
        if not key:
            print("  (no API key set in .env -- skipped)")
            continue
        try:
            models = lister(settings)
        except requests.RequestException as exc:
            print(f"  could not list models: {exc}")
            continue
        for m in models:
            print(f"  {'* ' if m == configured else '  '}{m}")
        if configured not in models:
            print(f"\n  !! Your configured model {configured!r} is NOT in this list.")
            print(f"     Pick one above and set it in .env.")
    print("\n(* = the model currently set in .env)")
    return 0


def preview(subject: str, body: str, indent: str = "    ") -> str:
    lines = [f"{indent}Subject: {subject}", indent + "-" * 60]
    lines += [indent + ln for ln in body.rstrip().split("\n")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_dry(
    contacts: List[Contact], profile: Profile, settings: Settings, log: logging.Logger
) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = settings.outbox_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    failures = 0
    for idx, c in enumerate(contacts, start=1):
        log.info(
            "[%d/%d] Generating for %s @ %s", idx, len(contacts), c.first_name, c.company
        )
        try:
            result = llm.generate_email(c.to_prompt_dict(), profile, settings)
        except llm.LLMError as exc:
            failures += 1
            log.error(
                "  FAILED to generate for %s @ %s: %s", c.first_name, c.company, exc
            )
            continue

        full_body = llm.compose(str(result["body"]), profile, settings.sender_name or None)
        subject = str(result["subject"])
        record = {
            "to": c.email,
            "first_name": c.first_name,
            "company": c.company,
            "title": c.title,
            "company_type": c.company_type,
            "subject": subject,
            "body": full_body,
            "provider": result["provider"],
            "body_word_count": result["word_count"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        generated.append(record)

        txt = out_dir / f"{idx:03d}_{slugify(c.company)}_{slugify(c.first_name)}.txt"
        txt.write_text(
            f"To: {c.first_name} <{c.email}>\nSubject: {subject}\n\n{full_body}",
            encoding="utf-8",
        )
        log.info("  saved -> %s", txt.relative_to(config.ROOT))
        log.debug("\n%s", preview(subject, full_body))

    json_path = out_dir / "emails.json"
    json_path.write_text(
        json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info("DRY RUN complete -- nothing was sent.")
    log.info("  generated : %d", len(generated))
    log.info("  failed    : %d", failures)
    log.info("  review    : %s", json_path.relative_to(config.ROOT))
    log.info("  then run  : python main.py --send")
    return 0 if failures == 0 else 1


def run_test(
    contacts: List[Contact], profile: Profile, settings: Settings, log: logging.Logger
) -> int:
    settings.require_smtp()
    if not contacts:
        log.error("No eligible contact to build a test email from.")
        return 1

    c = contacts[0]
    recipient = settings.test_recipient or settings.gmail_address
    log.info("TEST MODE: generating a sample email for %s @ %s", c.first_name, c.company)
    log.info("It goes ONLY to %s -- the real contact is not emailed.", recipient)

    result = llm.generate_email(c.to_prompt_dict(), profile, settings)
    full_body = llm.compose(str(result["body"]), profile, settings.sender_name or None)
    subject = f"[TEST] {result['subject']}"

    log.info("")
    log.info("%s", preview(subject, full_body))
    log.info("")

    with mailer.GmailMailer(
        settings.gmail_address,
        settings.gmail_app_password,
        settings.sender_name,
        settings.reply_to,
    ) as m:
        message_id = m.send(recipient, settings.sender_name, subject, full_body)

    log.info("Test email sent to %s (Message-ID %s)", recipient, message_id)
    log.info("Test sends are deliberately not written to sent_log.csv.")
    return 0


def run_send(
    contacts: List[Contact],
    profile: Profile,
    settings: Settings,
    sent_log: SentLog,
    args: argparse.Namespace,
    log: logging.Logger,
) -> int:
    settings.require_smtp()

    sent_today = sent_log.sent_today()
    remaining = settings.daily_send_cap - sent_today
    if remaining <= 0:
        log.warning(
            "Daily cap reached: %d/%d already sent today. Resume tomorrow.",
            sent_today,
            settings.daily_send_cap,
        )
        return 0

    batch = contacts[:remaining]
    log.info("Ready to send %d email(s) as %s.", len(batch), settings.gmail_address)
    log.info(
        "  daily cap : %d (%d sent today, %d left)",
        settings.daily_send_cap,
        sent_today,
        remaining,
    )
    log.info(
        "  delay     : %.0f-%.0fs between sends",
        settings.send_delay_min,
        settings.send_delay_max,
    )
    if len(contacts) > len(batch):
        log.info(
            "  deferred  : %d contact(s) beyond today's cap", len(contacts) - len(batch)
        )

    if not args.yes:
        try:
            answer = input(f"\nSend {len(batch)} real email(s) now? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in {"y", "yes"}:
            log.info("Aborted -- nothing sent.")
            return 0

    sent = 0
    failed = 0
    try:
        with mailer.GmailMailer(
            settings.gmail_address,
            settings.gmail_app_password,
            settings.sender_name,
            settings.reply_to,
        ) as m:
            for idx, c in enumerate(batch, start=1):
                log.info(
                    "[%d/%d] %s @ %s <%s>",
                    idx,
                    len(batch),
                    c.first_name,
                    c.company,
                    c.email,
                )

                try:
                    result = llm.generate_email(c.to_prompt_dict(), profile, settings)
                except llm.LLMError as exc:
                    failed += 1
                    log.error("  generation failed: %s", exc)
                    sent_log.record(c, tracker.STATUS_FAILED, error=f"LLM: {exc}")
                    continue

                full_body = llm.compose(
                    str(result["body"]), profile, settings.sender_name or None
                )
                subject = str(result["subject"])

                try:
                    message_id = m.send(c.email, c.first_name, subject, full_body)
                except mailer.MailerError as exc:
                    failed += 1
                    log.error("  send failed: %s", exc)
                    sent_log.record(
                        c,
                        tracker.STATUS_FAILED,
                        subject=subject,
                        provider=str(result["provider"]),
                        error=f"SMTP: {exc}",
                    )
                    continue

                sent += 1
                sent_log.record(
                    c,
                    tracker.STATUS_SENT,
                    subject=subject,
                    provider=str(result["provider"]),
                    message_id=message_id,
                )
                log.info("  SENT  subject=%r via %s", subject, result["provider"])

                if idx < len(batch):
                    pause = random.uniform(settings.send_delay_min, settings.send_delay_max)
                    log.info("  waiting %.1fs before the next send...", pause)
                    time.sleep(pause)
    except KeyboardInterrupt:
        log.warning(
            "Interrupted -- stopping. Progress is saved in %s",
            settings.sent_log_csv.name,
        )

    log.info("")
    log.info(
        "SEND complete.  sent=%d  failed=%d  tracked in %s",
        sent,
        failed,
        settings.sent_log_csv.name,
    )
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        settings = Settings.load()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    apply_overrides(settings, args)

    if args.list_models:
        try:
            settings.require_llm()
        except ConfigError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
        return list_models(settings)

    log_path = setup_logging(settings.log_dir, args.verbose)
    log = LOG
    mode = "SEND" if args.send else ("TEST" if args.test else "DRY RUN")
    log.info("AutoOutreach -- mode: %s", mode)
    log.info("Log file: %s", log_path)

    try:
        settings.require_llm()
        profile = Profile.load(settings.profile_json)
        contacts = tracker.load_contacts(settings.contacts_csv)
    except (ConfigError, tracker.ContactsError) as exc:
        log.error("%s", exc)
        return 2

    sent_log = SentLog(settings.sent_log_csv)
    selected = select_contacts(contacts, sent_log, args, log)
    log.info("%d of %d contact(s) eligible in this run.", len(selected), len(contacts))
    if not selected and not args.test:
        log.info("Nothing to do.")
        return 0

    try:
        if args.test:
            return run_test(selected or contacts, profile, settings, log)
        if args.send:
            return run_send(selected, profile, settings, sent_log, args, log)
        return run_dry(selected, profile, settings, log)
    except (ConfigError, llm.LLMError, mailer.MailerError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
