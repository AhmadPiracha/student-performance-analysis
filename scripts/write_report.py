"""Compatibility wrapper: delegate to the canonical report writer in scripts.report.

The repository historically contained two report writers (`scripts/report.py` and
`scripts/write_report.py`). To avoid duplication, `scripts/write_report.py` now
constructs a minimal config and calls `scripts.report.write_report`. This keeps
existing CLI calls that reference `write_report.py` working.
"""

from pathlib import Path
import scripts.report as canonical_report


def write_report(proc_dir):
    """Delegate to scripts.report.write_report using a minimal config.

    proc_dir may be a Path or string pointing to the processed data directory.
    """
    proc_dir = Path(proc_dir)
    cfg = {
        "paths": {
            "data_processed": str(proc_dir),
            "outputs_figures": str(proc_dir / "outputs_figures"),
            "outputs_reports": str(proc_dir),
        },
        "report": {
            "title": "Executive Function (EF) Integrated Analysis Report",
            "author": "Automated Script",
        }
    }
    return canonical_report.write_report(cfg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", required=True, help="Path to processed data folder")
    args = parser.parse_args()
    write_report(args.proc_dir)
