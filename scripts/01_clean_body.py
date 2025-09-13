#!/usr/bin/env python3
"""Simple cleaner for `data/processed/body_clean.csv` to fix header/data concat issues.

Strategy:
 - Find the header line (contains 'timestamp' and 'UserID' and 'condition').
 - If that header line also contains a data timestamp (e.g. '1970-01-01' immediately after),
   split it into header + a new data line.
 - Write cleaned file to the same path after creating a backup.

This is defensive and minimal; it avoids parsing the full CSV and only fixes the
common export issue observed in the repository.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SRC = PROC / "body_clean.csv"
BACKUP = PROC / "body_clean.csv.bak"


def clean_body_csv(src: Path, backup: Path):
    if not src.exists():
        print(f"Source not found: {src}")
        return False

    # Make backup
    if not backup.exists():
        backup.write_bytes(src.read_bytes())
        print(f"Backup written to {backup}")

    out_lines = []
    header_found = False

    # Pattern to detect a stray leading timestamp after the header
    ts_pat = re.compile(r"1970-01-01\s+00:00:00")
    # Also allow detecting an ISO 2025-like timestamp beginning a data row
    iso_ts = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")

    with src.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not header_found:
                low = line.lower()
                if "timestamp" in low and "userid" in low and "condition" in low:
                    # Found header line. Check if the header line contains a data timestamp afterward
                    # (export artifact). If so, split it.
                    # Try to split at the first occurrence of a timestamp-like token after 'condition'.
                    m = re.search(r"(1970-01-01\S*|\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
                    if m and m.start() > line.lower().find("condition"):
                        idx = m.start()
                        hdr = line[:idx].rstrip()
                        rest = line[idx:].lstrip()
                        out_lines.append(hdr + "\n")
                        out_lines.append(rest)
                    else:
                        out_lines.append(line)
                    header_found = True
                else:
                    out_lines.append(line)
            else:
                out_lines.append(line)

    # Write cleaned content back to src
    src.write_text("".join(out_lines), encoding="utf-8")
    print(f"Wrote cleaned file to {src}")
    return True


if __name__ == "__main__":
    ok = clean_body_csv(SRC, BACKUP)
    if not ok:
        raise SystemExit(1)