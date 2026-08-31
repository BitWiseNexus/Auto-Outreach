"""LLM-backed email generation.

Primary provider : Groq  (free tier, OpenAI-compatible chat completions API)
Fallback provider: Google Gemini (free tier)

Both are called over plain HTTPS with `requests`, so there are no vendor SDKs
to install and nothing here costs money. Each provider retries with
exponential backoff on rate limits / transient errors; if the primary is
exhausted the next provider is tried automatically.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import requests

from config import Profile, Settings

log = logging.getLogger("autooutreach.llm")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# HTTP statuses worth retrying: rate limit + transient server-side failures.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any failure to obtain a usable completion from a provider."""


class RetryableLLMError(LLMError):
    """Failure that is worth retrying (rate limit, 5xx, timeout)."""


class FatalProviderError(LLMError):
    """Misconfiguration -- bad key, missing model. Retrying cannot help, so
    the provider is abandoned immediately and the fallback is used instead."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You write short, professional cold outreach emails for a job seeker.

ABSOLUTE RULES:
1. Use ONLY the facts in the SENDER FACTS block. Never invent projects, companies, metrics, degrees, awards, years of experience, or mutual connections. If a detail is not in SENDER FACTS, it does not exist.
2. The skills list is a list of skills, NOT evidence of particular work. Never describe a project, system, product or accomplishment, and never attach a skill to a named employer (SENDER FACTS does not say what was built where). Write "I work with RAG and LangChain", never "at Accenture I built RAG pipelines".
3. Never claim to have used the recipient's product, never claim to have been following the company, and never fabricate admiration for specific work.
4. The body must be under {max_words} words. Plain text only: no markdown, no bold, no bullet symbols, no emojis, no links, and no placeholders such as [Company]. Use straight quotes and plain hyphens, never typographic ones.
5. Separate the greeting, every paragraph, and the closing line with a blank line, so the plain-text email is readable.
6. Do NOT write a signature, a name, or any links at the end. Stop after the closing line. A signature block is appended separately by the program.
7. Open with "Hi <first name>," on its own line, using the first name only.
8. Respect the tense in SENDER FACTS: describe a completed role in the past tense ("interned at X") and never present it as ongoing. Never state or imply a parenthetical note from SENDER FACTS verbatim.
9. Tone: direct, respectful, specific, zero filler. No "I hope this email finds you well", no "I am reaching out to express my keen interest", no buzzwords, no exclamation marks.

CONTENT: The body MUST name the company at least once, and the subject line MUST contain the company name -- a mail that would read identically to any other company is a failure. Write one short opening line naming the company and connecting it to the sender's actual background; one or two lines of concrete relevant skills drawn from SENDER FACTS; then a polite question about current or upcoming Software Engineering / GenAI openings suitable for a final-year student graduating soon; and a one-line close pointing to the resume and links below.

Return ONLY a JSON object, with no prose around it:
{{"subject": "<subject line, under 9 words>", "body": "<email body, newlines as \\n>"}}"""


def build_user_prompt(profile: Profile, contact: Dict[str, str]) -> str:
    return (
        "SENDER FACTS (the only facts you may use):\n"
        f"{profile.facts_block()}\n\n"
        "RECIPIENT:\n"
        f"- First name: {contact.get('first_name', '')}\n"
        f"- Company: {contact.get('company', '')}\n"
        f"- Their title: {contact.get('title', '')}\n"
        f"- Company type: {contact.get('company_type', '')}\n\n"
        "Write the email. Reference the company by name and let the company "
        "type shape which part of the sender's background you lead with "
        "(an AI / conversational-AI company -> lead with the GenAI, RAG and "
        "agentic-workflow work; a SaaS or data-infrastructure company -> lead "
        "with the full-stack and backend work). The recipient works in talent "
        "acquisition / HR, so ask them directly about openings rather than "
        "about technical details. Return the JSON object only."
    )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
@dataclass
class Provider:
    name: str
    call: Callable[[str, str], str]


def _groq_call(settings: Settings) -> Callable[[str, str], str]:
    # Some Groq models (gpt-oss in particular) intermittently fail Groq's own
    # server-side JSON validator even when the prompt is fine. Our parser
    # tolerates loose JSON anyway, so after the first such failure we stop
    # asking for strict JSON mode for the rest of the run.
    state = {"strict_json": True}

    def call(system: str, user: str) -> str:
        payload: Dict[str, object] = {
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
            # Roomy on purpose: a reasoning model (gpt-oss, qwen3) spends part
            # of this budget thinking before it emits any content, and a budget
            # that runs out leaves `content` empty or the JSON half-written.
            "max_tokens": 2048,
        }
        if state["strict_json"]:
            payload["response_format"] = {"type": "json_object"}
        # There is nothing to reason about in a 100-word email, so keep the
        # thinking minimal and out of the reply.
        if "gpt-oss" in settings.groq_model:
            payload["reasoning_effort"] = "low"

        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RetryableLLMError(f"Groq network error: {exc}") from exc

        if resp.status_code in RETRYABLE_STATUS:
            raise RetryableLLMError(f"Groq HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code in (401, 403, 404):
            raise FatalProviderError(
                f"Groq HTTP {resp.status_code}: {resp.text[:300]}\n"
                "  -> check GROQ_API_KEY and GROQ_MODEL in .env. "
                "Run 'python main.py --list-models' to see what your key can use."
            )
        if resp.status_code >= 400:
            # json_validate_failed = the model broke its own JSON mode. That is
            # a dice roll, not a config problem, so it is worth another attempt.
            if "json_validate_failed" in resp.text:
                if state["strict_json"]:
                    state["strict_json"] = False
                    log.warning(
                        "Groq's JSON mode rejected %s output; disabling strict "
                        "JSON mode for the rest of this run and retrying.",
                        settings.groq_model,
                    )
                raise RetryableLLMError(f"Groq returned malformed JSON: {resp.text[:200]}")
            raise LLMError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            choice = resp.json()["choices"][0]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected Groq response: {resp.text[:300]}") from exc

        message = choice.get("message", {})
        text = message.get("content") or ""
        if not text.strip():
            # A reasoning model can put everything in `reasoning` and leave
            # `content` empty; the JSON is still in there to be salvaged.
            text = message.get("reasoning") or ""
        if not text.strip():
            raise RetryableLLMError(
                f"Groq returned empty content "
                f"(finish_reason={choice.get('finish_reason')!r}). "
                "The model may have spent its budget on reasoning."
            )
        return text

    return call


def _gemini_call(settings: Settings) -> Callable[[str, str], str]:
    def call(system: str, user: str) -> str:
        url = GEMINI_URL.format(model=settings.gemini_model)
        generation_config: Dict[str, object] = {
            "temperature": 0.7,
            # Generous: an email is ~250 tokens, but a Gemini 2.5 model spends
            # part of the budget on internal thinking before it emits anything,
            # and a budget that runs out mid-string yields truncated JSON.
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        }
        # Gemini 2.5 models think by default. There is nothing to reason about
        # in a 100-word email, and thinking tokens are what caused the
        # truncation above, so switch it off where the model supports it.
        if "2.5" in settings.gemini_model and "pro" not in settings.gemini_model:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        try:
            resp = requests.post(
                url,
                headers={
                    "x-goog-api-key": settings.gemini_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": generation_config,
                },
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RetryableLLMError(f"Gemini network error: {exc}") from exc

        if resp.status_code in RETRYABLE_STATUS:
            raise RetryableLLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code in (401, 403, 404):
            raise FatalProviderError(
                f"Gemini HTTP {resp.status_code}: {resp.text[:300]}\n"
                "  -> check GEMINI_API_KEY and GEMINI_MODEL in .env. "
                "Run 'python main.py --list-models' to see what your key can use."
            )
        if resp.status_code >= 400:
            raise LLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise LLMError(f"Unexpected Gemini response: {resp.text[:300]}") from exc

        if not payload.get("candidates"):
            # Safety filter or a prompt-level block: no candidate at all.
            raise LLMError(f"Gemini returned no candidates: {resp.text[:300]}")

        candidate = payload["candidates"][0]
        finish = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts", []) or []
        text = "".join(p.get("text", "") for p in parts)

        if finish == "MAX_TOKENS" and not text.rstrip().endswith("}"):
            raise RetryableLLMError(
                "Gemini hit maxOutputTokens and returned truncated JSON. "
                "Raise maxOutputTokens or use a non-thinking model."
            )
        if not text:
            raise LLMError(
                f"Gemini returned no text (finishReason={finish!r}): {resp.text[:300]}"
            )
        return text

    return call


# ---------------------------------------------------------------------------
# Model discovery -- so a decommissioned model name is a 10-second fix.
# ---------------------------------------------------------------------------
def list_groq_models(settings: Settings) -> List[str]:
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json().get("data", []))


def list_gemini_models(settings: Settings) -> List[str]:
    resp = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": settings.gemini_api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return sorted(
        m["name"].removeprefix("models/")
        for m in resp.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    )


def available_providers(settings: Settings) -> List[Provider]:
    """Groq first, Gemini second. Providers without a key are skipped."""
    providers: List[Provider] = []
    if settings.groq_api_key:
        providers.append(Provider(f"groq:{settings.groq_model}", _groq_call(settings)))
    if settings.gemini_api_key:
        providers.append(
            Provider(f"gemini:{settings.gemini_model}", _gemini_call(settings))
        )
    return providers


# ---------------------------------------------------------------------------
# Response parsing & sanitising
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*`{3}(?:json)?\s*|\s*`{3}\s*$", re.IGNORECASE)

# Sign-off lines the model sometimes appends despite being told not to.
_SIGNOFF_RE = re.compile(
    r"^\s*(best regards|kind regards|warm regards|regards|best|sincerely|"
    r"thanks|thank you|cheers|yours sincerely|yours truly)\s*[,.]?\s*$",
    re.IGNORECASE,
)


def _extract_json(text: str) -> Dict[str, str]:
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        data = json.loads(cleaned)
    except ValueError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise LLMError(f"Model did not return JSON: {text[:200]}")
        try:
            data = json.loads(match.group(0))
        except ValueError as exc:
            raise LLMError(f"Model returned invalid JSON: {text[:200]}") from exc
    if not isinstance(data, dict) or "subject" not in data or "body" not in data:
        raise LLMError(f"Model JSON missing subject/body: {text[:200]}")
    return {"subject": str(data["subject"]), "body": str(data["body"])}


# Typographic characters LLMs like to emit. They are legal in a UTF-8 email but
# render inconsistently in plain-text clients, and a non-breaking hyphen in
# "final-year" looks like a glitch, so fold them back to ASCII.
_UNICODE_FIXES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "--",
    "―": "-", "−": "-",
    "…": "...", "•": "-", " ": " ", " ": " ",
    " ": " ", "​": "", "﻿": "",
}


def normalize_text(text: str) -> str:
    for bad, good in _UNICODE_FIXES.items():
        text = text.replace(bad, good)
    return text


def mentions_company(text: str, company: str) -> bool:
    """Is the company actually named in this text?

    Matches on the distinctive first token so "Wingify (VWO)" is satisfied by
    "Wingify", and punctuation/case differences ("Observe.AI" vs "Observe AI")
    do not cause a false negative.
    """
    if not company:
        return True

    def squash(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    haystack = squash(text)
    tokens = [t for t in re.split(r"[\s(),./|-]+", company) if len(squash(t)) >= 3]
    if not tokens:
        tokens = [company]
    return squash(tokens[0]) in haystack


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _strip_trailing_signature(body: str, profile: Profile) -> str:
    """Remove any name / link / sign-off tail the model added, so the only
    signature in the email is the deterministic one from profile.json."""
    lines = normalize_text(body).rstrip().split("\n")
    link_values = [v.lower() for v in profile.links.values()]
    first_name = profile.name.split()[0].lower()

    def is_tail_junk(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        low = s.lower()
        if s in {"--", "-", "—"}:
            return True
        if low in {profile.name.lower(), first_name}:
            return True
        if "http://" in low or "https://" in low or "www." in low:
            return True
        if any(url in low for url in link_values):
            return True
        if _SIGNOFF_RE.match(s):
            return True
        # "Resume: ...", "GitHub: ..." style lines
        if re.match(r"^(resume|cv|github|linkedin|portfolio)\s*[:\-]", low):
            return True
        return False

    while lines and is_tail_junk(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _sanitize_subject(subject: str) -> str:
    # A subject is a single header line: collapse any stray newlines.
    subject = normalize_text(subject).strip().strip('"').strip()
    subject = re.sub(r"\s+", " ", subject.replace("\n", " ").replace("\r", " "))
    return subject[:120]


def compose(body: str, profile: Profile, sender_name: str | None = None) -> str:
    """Attach the deterministic signature block to a generated body."""
    return f"{body.rstrip()}\n\n{profile.signature(sender_name)}\n"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_email(
    contact: Dict[str, str],
    profile: Profile,
    settings: Settings,
    providers: List[Provider] | None = None,
) -> Dict[str, object]:
    """Generate {subject, body, provider, word_count} for one contact.

    Tries each provider in order; within a provider, retries with exponential
    backoff + jitter on rate limits and transient errors.
    """
    providers = providers if providers is not None else available_providers(settings)
    if not providers:
        raise LLMError(
            "No LLM provider configured -- set GROQ_API_KEY and/or GEMINI_API_KEY."
        )

    system = SYSTEM_PROMPT.format(max_words=settings.max_body_words)
    base_user = build_user_prompt(profile, contact)
    last_error: Exception | None = None

    for provider in providers:
        user = base_user
        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                raw = provider.call(system, user)
                parsed = _extract_json(raw)
                subject = _sanitize_subject(parsed["subject"])
                body = _strip_trailing_signature(parsed["body"].strip(), profile)
                if not subject or not body:
                    raise LLMError("Model returned an empty subject or body.")

                words = word_count(body)
                if words > settings.max_body_words:
                    if attempt < settings.llm_max_retries:
                        log.warning(
                            "%s: body was %d words (limit %d); retrying tighter.",
                            provider.name,
                            words,
                            settings.max_body_words,
                        )
                        user = (
                            base_user
                            + f"\n\nYour previous attempt was {words} words. "
                            f"Rewrite it in under {settings.max_body_words} words."
                        )
                        continue
                    log.warning(
                        "%s: body is %d words, over the %d-word target -- keeping it.",
                        provider.name,
                        words,
                        settings.max_body_words,
                    )

                # A mail that never names the company is not outreach, it is a
                # form letter -- reject it and say why, rather than sending it.
                company = str(contact.get("company", ""))
                if not mentions_company(body, company):
                    if attempt < settings.llm_max_retries:
                        log.warning(
                            "%s: body never mentions %s; regenerating.",
                            provider.name,
                            company,
                        )
                        user = (
                            base_user
                            + f"\n\nYour previous attempt never mentioned "
                            f"{company} in the body. Rewrite it so the opening "
                            f"line names {company} explicitly."
                        )
                        continue
                    raise LLMError(
                        f"Model would not mention {company} in the body after "
                        f"{attempt} attempts."
                    )
                if not mentions_company(subject, company):
                    log.warning(
                        "%s: subject does not name %s (body does) -- accepting.",
                        provider.name,
                        company,
                    )

                log.info(
                    "Generated email via %s (%d words) for %s @ %s",
                    provider.name,
                    words,
                    contact.get("first_name", "?"),
                    contact.get("company", "?"),
                )
                return {
                    "subject": subject,
                    "body": body,
                    "provider": provider.name,
                    "word_count": words,
                }

            except FatalProviderError as exc:
                # Bad key or missing model: more attempts cannot help.
                last_error = exc
                log.error("%s is misconfigured: %s", provider.name, exc)
                break

            except RetryableLLMError as exc:
                last_error = exc
                if attempt == settings.llm_max_retries:
                    log.warning(
                        "%s exhausted %d attempts: %s", provider.name, attempt, exc
                    )
                    break
                sleep_for = settings.llm_backoff_base**attempt + random.uniform(0, 1)
                log.warning(
                    "%s attempt %d/%d failed (%s). Backing off %.1fs.",
                    provider.name,
                    attempt,
                    settings.llm_max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)

            except LLMError as exc:
                # Bad output / auth / bad model name: try once more, then move
                # on to the next provider rather than burning the retry budget.
                last_error = exc
                log.warning(
                    "%s attempt %d/%d error: %s",
                    provider.name,
                    attempt,
                    settings.llm_max_retries,
                    exc,
                )
                if attempt >= 2:
                    break
                time.sleep(1.0)

        if provider is not providers[-1]:
            log.warning("Falling back to the next LLM provider after %s.", provider.name)

    raise LLMError(f"All LLM providers failed. Last error: {last_error}")
