"""packs the run folders of an experiment into one zip under exports, without their large files, so the tables of
many runs come down from Drive in one small download.

    python scripts/export_runs.py e11_anoshift_prevalence
    python scripts/export_runs.py e12_litnet_prevalence --from 20261013-224115_8502bf0 --max-mb 20

every run folder of the experiment at or after --from goes in with its config, meta, log, tables, summaries and
figures. a file above --max-mb, such as the per flow scores, is left out and listed with its size in left_out.txt
inside the run folder of the zip. scripts/import_runs.py imports the zip as it imports a Drive download, and the
local copy of such a run then lacks the files listed there.
"""

import argparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from forge import paths, runs


def export(experiment: str, out: Path, max_mb: float, first: str | None = None, root=None, log=print) -> Path:
    folders = [f for f in runs.run_dirs(experiment, root) if first is None or f.name >= first]
    if not folders:
        raise SystemExit(f"no run of {experiment} to export")
    out.parent.mkdir(parents=True, exist_ok=True)
    limit = max_mb * 1024 * 1024
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            left_out = []
            for path in sorted(p for p in folder.rglob("*") if p.is_file()):
                size = path.stat().st_size
                if size > limit:
                    left_out.append(f"{path.relative_to(folder)} {size} bytes")
                    continue
                z.write(path, f"{experiment}/{folder.name}/{path.relative_to(folder)}")
            if left_out:
                z.writestr(f"{experiment}/{folder.name}/left_out.txt", "\n".join(left_out) + "\n")
            log(f"packed {folder.name}, {len(left_out)} files above {max_mb} MB left out")
    log(f"wrote {out}, {out.stat().st_size / 1024 / 1024:.1f} MB, {len(folders)} runs")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--from", dest="first", default=None, metavar="RUN_ID", help="the earliest run to pack")
    parser.add_argument("--max-mb", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out = args.out or paths.storage_root() / "exports" / f"{args.experiment}_{stamp}.zip"
    export(args.experiment, out, args.max_mb, args.first)


if __name__ == "__main__":
    main()
