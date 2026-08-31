"""Gmail SMTP sending (free -- no paid email API anywhere).

Uses smtplib + a Gmail App Password over implicit TLS (port 465). Each
contact gets their own individually addressed message; there is deliberately
no CC/BCC batching, which is both better outreach and much less likely to be
flagged as bulk mail.

NOTE ON PAID ALTERNATIVES: transactional providers (SendGrid, Mailgun, SES,
Postmark) would normally be used at scale and all charge beyond a trial.
None of them are used here -- plain Gmail SMTP is enough for ~40 mails/day,
which is also the safe limit for a personal Gmail account.
"""

from __future__ import annotations

import logging
import re
import smtplib
import ssl
from email.header import Header
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import Optional

log = logging.getLogger("autooutreach.mailer")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # implicit TLS

# Deliberately conservative: one @, a dotted domain, no spaces or commas.
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


class MailerError(RuntimeError):
    """SMTP connection / authentication / send failure."""


def is_valid_email(address: str) -> bool:
    if not address:
        return False
    address = address.strip()
    if len(address) > 254 or ".." in address:
        return False
    return bool(EMAIL_RE.match(address))


def build_message(
    *,
    sender_name: str,
    sender_email: str,
    to_email: str,
    to_name: str,
    subject: str,
    body: str,
    reply_to: Optional[str] = None,
) -> EmailMessage:
    """Build a plain-text, individually addressed message."""
    msg = EmailMessage()
    msg["From"] = formataddr((str(Header(sender_name, "utf-8")), sender_email))
    msg["To"] = formataddr((str(Header(to_name, "utf-8")), to_email)) if to_name else to_email
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender_email.split("@")[-1])
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(body)
    return msg


class GmailMailer:
    """Context manager holding one authenticated SMTP connection for a run."""

    def __init__(self, address: str, app_password: str, sender_name: str,
                 reply_to: Optional[str] = None) -> None:
        self.address = address
        self.app_password = app_password
        self.sender_name = sender_name or address
        self.reply_to = reply_to or address
        self._smtp: Optional[smtplib.SMTP_SSL] = None

    # -- connection -------------------------------------------------------
    def connect(self) -> None:
        context = ssl.create_default_context()
        try:
            self._smtp = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=45)
            self._smtp.login(self.address, self.app_password)
        except smtplib.SMTPAuthenticationError as exc:
            raise MailerError(
                "Gmail rejected the login. Use a 16-character App Password "
                "(https://myaccount.google.com/apppasswords), not your normal "
                f"Gmail password. Server said: {exc}"
            ) from exc
        except (smtplib.SMTPException, OSError) as exc:
            raise MailerError(f"Could not connect to Gmail SMTP: {exc}") from exc
        log.info("Connected to %s:%d as %s", SMTP_HOST, SMTP_PORT, self.address)

    def close(self) -> None:
        if self._smtp is not None:
            try:
                self._smtp.quit()
            except smtplib.SMTPException:
                pass
            finally:
                self._smtp = None

    def __enter__(self) -> "GmailMailer":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- sending ----------------------------------------------------------
    def send(self, to_email: str, to_name: str, subject: str, body: str) -> str:
        """Send one email. Returns the Message-ID. Raises MailerError."""
        if not is_valid_email(to_email):
            raise MailerError(f"Invalid recipient address: {to_email!r}")
        if self._smtp is None:
            self.connect()

        msg = build_message(
            sender_name=self.sender_name,
            sender_email=self.address,
            to_email=to_email.strip(),
            to_name=to_name,
            subject=subject,
            body=body,
            reply_to=self.reply_to,
        )
        try:
            self._smtp.send_message(msg)
        except smtplib.SMTPServerDisconnected:
            # Gmail drops idle connections; reconnect once and retry.
            log.warning("SMTP connection dropped -- reconnecting and retrying once.")
            self._smtp = None
            self.connect()
            try:
                self._smtp.send_message(msg)
            except smtplib.SMTPException as exc:
                raise MailerError(f"Send failed after reconnect: {exc}") from exc
        except smtplib.SMTPException as exc:
            raise MailerError(f"Send failed: {exc}") from exc

        return msg["Message-ID"]

    def verify(self) -> None:
        """Cheap credential check used before a real send run."""
        self.connect()
        self.close()
