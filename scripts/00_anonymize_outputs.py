"""Anonymize output delivery CSVs by replacing UUID user_id with Student_X labels.

Creates files in outputs/datasets/ with suffix `_studentlabels.csv`.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAPPING = ROOT / 'data' / 'processed' / 'id_mapping_secure.csv'
INPUT_DIR = ROOT / 'outputs' / 'datasets'
FILES = [
    'games_summary.csv',
    'games_emotion_summary.csv',
    'games_emotion_body_summary.csv',
]


def load_mapping(p: Path):
    m = {}
    with p.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            m[r['user_id']] = r['label']
    return m


def anonymize_file(src: Path, dst: Path, mapping: dict):
    with src.open() as inf, dst.open('w', newline='') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        headers = next(reader)
        # find user_id column index
        try:
            uid_idx = headers.index('user_id')
        except ValueError:
            uid_idx = None
        writer.writerow(headers)
        for row in reader:
            if uid_idx is not None and row[uid_idx] in mapping:
                row[uid_idx] = mapping[row[uid_idx]]
            writer.writerow(row)


def main():
    mapping = load_mapping(MAPPING)
    for fn in FILES:
        src = INPUT_DIR / fn
        if not src.exists():
            print(f"Skipping missing file: {src}")
            continue
        dst = INPUT_DIR / (src.stem + '_studentlabels.csv')
        anonymize_file(src, dst, mapping)
        print(f"Wrote: {dst}")


if __name__ == '__main__':
    main()
