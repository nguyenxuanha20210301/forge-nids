"""moves runs downloaded from Drive into the local mirror at drive/runs.

Drive packs a downloaded folder into one or more zip files, and splits large downloads across several. pass all of them at once so the parts are merged before anything is moved.

    python scripts/import_runs.py ~/Downloads/e00_smoke-20261010T091500Z-001.zip
    python scripts/import_runs.py ~/Downloads/*.zip --delete

a run is any folder holding a meta.json. runs that already exist locally are left untouched.
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from forge import paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zips", nargs="+", type=Path)
    parser.add_argument("--delete", action="store_true", help="remove the zip files after a clean import")
    args = parser.parse_args()

    target = paths.runs_dir()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for z in args.zips:
            with zipfile.ZipFile(z) as f:
                f.extractall(tmp)
        found = sorted(tmp.rglob("meta.json"))
        if not found:
            raise SystemExit("no run folder found, a run folder holds a meta.json")
        for meta_path in found:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dest = target / meta["experiment"] / meta["run_id"]
            if dest.exists():
                print(f"skipped {meta['experiment']}/{meta['run_id']}, already present")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(meta_path.parent), dest)
            print(f"imported {meta['experiment']}/{meta['run_id']}, status {meta.get('status')}")
    if args.delete:
        for z in args.zips:
            z.unlink()
        print(f"deleted {len(args.zips)} zip files")


if __name__ == "__main__":
    main()
