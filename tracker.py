"""Contact loading and sent-state tracking.

Reads the contacts CSV (tolerating header variations) and maintains
`sent_log.csv`, which is the single source of truth for what has already
gone out. Reruns skip anything already marked SENT, and the daily cap is
computed from today's SENT rows in that same file.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

log = logging.getLogger("autooutreach.tracker")

SENT_LOG_FIELDS = [
    "Email",
    "First Name",
    "Company",
    "Title",
    "Status",
    "SentAt",
    "Subject",
    "Provider",
    "MessageID",
    "Error",
]

STATUS_SENT = "SENT"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
STATUS_DRYRUN = "DRY_RUN"


# ---------------------------------------------------------------------------
# Contacts CSV
# ---------------------------------------------------------------------------
@dataclass
class Contact:
    first_name: str
    company: str
    title: str
    email: str
    company_type: str
    salary_notes: str  # for the user's own reference only -- never sent to the LLM
    row_number: int

    def to_prompt_dict(self) -> Dict[str, str]:
        """Only the fields the LLM is allowed to see (salary notes excluded)."""
        return {
            "first_name": self.first_name,
            "company": self.company,
            "title": self.title,
            "company_type": self.company_type,
        }


def _norm(header: str) -> str:
    return re.sub(r"[^a-z]", "", (header or "").lower())


# Accepts "First Name", "first_name", "Salary Notes (India, SWE)", etc.
_FIELD_ALIASES = {
    "first_name": {"firstname", "first", "name", "contactfirstname"},
    "company": {"company", "companyname", "organisation", "organization"},
    "title": {"title", "role", "designation", "jobtitle"},
    "email": {"email", "emailaddress", "mail"},
    "company_type": {"companytype", "type", "companycategory"},
    "salary_notes": {"salarynotes", "notes", "salary"},
}


def _map_headers(fieldnames: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for header in fieldnames or []:
        key = _norm(header)
        for field, aliases in _FIELD_ALIASES.items():
            if field in mapping:
                continue
            if key in aliases or any(key.startswith(a) for a in aliases):
                mapping[field] = header
                break
    return mapping


class ContactsError(RuntimeError):
    pass


def load_contacts(csv_path: Path) -> List[Contact]:
    if not csv_path.exists():
        raise ContactsError(f"Contacts CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        mapping = _map_headers(reader.fieldnames or [])
        required = ("first_name", "company", "title", "email", "company_type")
        missing = [f for f in required if f not in mapping]
        if missing:
            raise ContactsError(
                f"{csv_path.name} is missing column(s) for: {', '.join(missing)}. "
                f"Found headers: {reader.fieldnames}"
            )

        contacts: List[Contact] = []
        for i, row in enumerate(reader, start=2):  # row 1 is the header
            def get(field: str) -> str:
                col = mapping.get(field)
                return (row.get(col) or "").strip() if col else ""

            if not any(get(f) for f in required):
                continue  # blank line
            contacts.append(
                Contact(
                    first_name=get("first_name"),
                    company=get("company"),
                    title=get("title"),
                    email=get("email"),
                    company_type=get("company_type"),
                    salary_notes=get("salary_notes"),
                    row_number=i,
                )
            )
    log.info("Loaded %d contacts from %s", len(contacts), csv_path.name)
    return contacts


# ---------------------------------------------------------------------------
# Sent log
# ---------------------------------------------------------------------------
class SentLog:
    """Append-only CSV of every send attempt."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: List[Dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8-sig", newline="") as fh:
            self.rows = [dict(r) for r in csv.DictReader(fh)]

    # -- queries ----------------------------------------------------------
    def already_sent(self) -> Set[str]:
        return {
            (r.get("Email") or "").strip().lower()
            for r in self.rows
            if (r.get("Status") or "").upper() == STATUS_SENT
        }

    def sent_today(self, today: Optional[str] = None) -> int:
        today = today or datetime.now().strftime("%Y-%m-%d")
        return sum(
            1
            for r in self.rows
            if (r.get("Status") or "").upper() == STATUS_SENT
            and (r.get("SentAt") or "").startswith(today)
        )

    # -- writes -----------------------------------------------------------
    def record(
        self,
        contact: Contact,
        status: str,
        subject: str = "",
        provider: str = "",
        message_id: str = "",
        error: str = "",
    ) -> None:
        row = {
            "Email": contact.email,
            "First Name": contact.first_name,
            "Company": contact.company,
            "Title": contact.title,
            "Status": status,
            "SentAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Subject": subject,
            "Provider": provider,
            "MessageID": message_id,
            "Error": error.replace("\n", " ")[:500],
        }
        self.rows.append(row)
        self._append(row)

    def _append(self, row: Dict[str, str]) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SENT_LOG_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in SENT_LOG_FIELDS})
