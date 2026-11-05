"""Shared helpers: safe console printing, raw-file loading, text report writer."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import config

# Text columns that must be stripped of accidental spaces before we compare them.
_TEXT_COLUMNS = {
    "train": ["family"],
    "test": ["family"],
    "stores": ["city", "state", "type"],
    "holidays": ["type", "locale", "locale_name", "description"],
}


def setup_console() -> None:
    """Make printing safe on Windows consoles (no UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class Report:
    """Prints every line AND keeps it so it can be saved to a text file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.add("")
        self.add("=" * 78)
        self.add(title)
        self.add("=" * 78)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_raw(name: str) -> pd.DataFrame:
    """Read one raw CSV exactly as it is, only making types reliable:
    dates -> datetime64[ns], text columns stripped, holidays.transferred -> bool.
    No rows are added, removed or changed.
    """
    path = config.RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").astype("datetime64[ns]")

    for col in _TEXT_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = df[col].str.strip()

    if name == "holidays":
        flag = df["transferred"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        if flag.isna().any():
            raise ValueError("holidays_events.csv: 'transferred' has values other than True/False")
        df["transferred"] = flag.astype(bool)

    return df


def missing_required_files() -> list[Path]:
    return [config.RAW_FILES[n] for n in config.REQUIRED_RAW if not config.RAW_FILES[n].exists()]