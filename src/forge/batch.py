"""one config run many times as separate processes, once per seed or once per variant of a sweep, each run in its own
folder.

every run is a child process of scripts/run.py, so a failed run stops nothing else and its memory goes with it. a
seed or variant whose resolved config already has a finished run in the runs folder is skipped, so a batch that a
Colab session cut short picks up where it stopped when it is started again.
"""

import subprocess
import sys
import time
from pathlib import Path

import yaml

from forge import paths, runs
from forge.config import load_config

# a config key that describes the whole batch and not the run, left out when two configs are compared
BATCH_KEYS = ("sweep",)


def parse_seeds(items) -> list[int]:
    # 1-19 is the range 1 to 19 and 3 one seed, in the order given, each seed once
    out = []
    for item in items:
        text = str(item)
        if "-" in text:
            lo, hi = text.split("-", 1)
            if int(hi) < int(lo):
                raise ValueError(f"the seed range {text} ends before it starts")
            out += range(int(lo), int(hi) + 1)
        else:
            out.append(int(text))
    return list(dict.fromkeys(out))


def override(key: str, value) -> str:
    # a key=value item for scripts/run.py, lists and mappings in yaml flow style on one line
    text = yaml.safe_dump(value, default_flow_style=True, width=10 ** 6, sort_keys=False).strip()
    if text.endswith("..."):
        text = text[:-3].strip()
    return f"{key}={text}"


def overrides(sets: dict) -> list[str]:
    return [override(k, v) for k, v in sets.items()]


def comparable(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in BATCH_KEYS}


def finished_run(cfg: dict, root=None) -> Path | None:
    # the latest finished run of the experiment whose saved config equals cfg, none when there is no such run
    for run_dir in reversed(runs.run_dirs(cfg["experiment"]["name"], root)):
        if runs.load_meta(run_dir).get("status") == "done" and \
                comparable(runs.load_run_config(run_dir)) == comparable(cfg):
            return run_dir
    return None


def run_once(config, sets, log=print) -> dict:
    # scripts/run.py in a child process, its lines passed on as they come, the run id read from its first line
    cmd = [sys.executable, str(paths.repo_root() / "scripts" / "run.py"), str(config)]
    if sets:
        cmd += ["--set", *sets]
    started = time.time()
    run_id = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log(line.rstrip("\n"))
        if run_id is None and ", run " in line:
            run_id = line.rsplit(", run ", 1)[1].strip()
    code = proc.wait()
    return {"run_id": run_id, "status": "done" if code == 0 else "failed", "returncode": code,
            "seconds": round(time.time() - started, 1)}


def run_batch(config, items, rerun: bool = False, log=print) -> list[dict]:
    """runs config once per item, each item a label and its list of key=value overrides, and returns one row per item
    with its run id and status. a finished run with the same resolved config is reused unless rerun is set."""
    config = Path(config)
    rows = []
    for label, sets in items:
        cfg = load_config(config, sets)
        done = None if rerun else finished_run(cfg)
        if done is not None:
            log(f"{label}, skipped, finished run {done.name} has the same config")
            rows.append({"label": label, "run_id": done.name, "status": "skipped", "seconds": 0.0})
            continue
        log(f"{label}, running {config.name} with " + (" ".join(sets) if sets else "no override"))
        out = run_once(config, sets, log)
        rows.append({"label": label, **out})
    failed = sum(r["status"] == "failed" for r in rows)
    skipped = sum(r["status"] == "skipped" for r in rows)
    log(f"batch of {len(rows)} runs done, {failed} failed, {skipped} skipped, "
        f"{sum(r['seconds'] for r in rows):.1f} s in all")
    for r in rows:
        log(f"  {r['label']}, run {r['run_id']}, {r['status']}, {r['seconds']:.1f} s")
    return rows
