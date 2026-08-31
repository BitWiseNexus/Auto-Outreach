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


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You write short, professional cold outreach emails for a job seeker.

ABSOLUTE RULES:
1. Use ONLY the facts in the SENDER FACTS block. Never invent projects, companies, metrics, degrees, awards, years of experience, or mutual connections. If a detail is not in SENDER FACTS, it does not exist.
2. Never claim to have used the recipient's product, never claim to have been following the company, and never fabricate admiration for specific work.
3. The body must be under {max_words} words. Plain text only: no markdown, no bold, no bullet symbols, no emojis, no links, and no placeholders such as [Company].
4. Do NOT write a signature, a name, or any links at the end. Stop after the closing line. A signature block is appended separately by the program.
5. Address the recipient by first name only.
6. Respect the tense in SENDER FACTS: describe a completed role in the past tense ("interned at X") and never present it as ongoing. Never state or imply a parenthetical note from SENDER FACTS verbatim.
7. Tone: direct, respectful, specific, zero filler. No "I hope this email finds you well", no "I am reaching out to express my keen interest", no buzzwords, no exclamation marks.

CONTENT: one short opening line naming the company and connecting it to the sender's actual background; one or two lines of concrete relevant skills drawn from SENDER FACTS; then a polite question about current or upcoming Software Engineering / GenAI openings suitable for a final-year student graduating soon; and a one-line close pointing to the resume and links below.

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
    def call(system: str, user: str) -> str:
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.groq_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 700,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RetryableLLMError(f"Groq network error: {exc}") from exc

        if resp.status_code in RETRYABLE_STATUS:
            raise RetryableLLMError(f"Groq HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected Groq response: {resp.text[:300]}") from exc

    return call


def _gemini_call(settings: Settings) -> Callable[[str, str], str]:
    def call(system: str, user: str) -> str:
        url = GEMINI_URL.format(model=settings.gemini_model)
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
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 700,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RetryableLLMError(f"Gemini network error: {exc}") from exc

        if resp.status_code in RETRYABLE_STATUS:
            raise RetryableLLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            parts = resp.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected Gemini response: {resp.text[:300]}") from exc

    return call


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


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _strip_trailing_signature(body: str, profile: Profile) -> str:
    """Remove any name / link / sign-off tail the model added, so the only
    signature in the email is the deterministic one from profile.json."""
    lines = body.rstrip().split("\n")
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
    subject = subject.strip().strip('"').strip()
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
