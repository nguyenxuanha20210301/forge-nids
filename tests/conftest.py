"""tiny archives in the layout of the published LITNET-2020 files, written under a temporary storage root."""

import hashlib
import random
import zipfile
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from forge.config import load_config
from forge.paths import configs_dir


def litnet_line(row_id, te, attack_t="none", attack_a=0, td=0.0, pr="TCP", flags="....S.", sp=40000, dp=80):
    # 85 quoted fields as in the published csv files, rule columns filled the way the authors fill them
    ts = te - timedelta(seconds=int(td))
    rules = ["normal"] * 19 if attack_a == 0 else ["blank"] * 19

    def parts(t):
        return [f"{t:%Y}", f"{t:%m}", f"{t:%d}", f"{t:%H}", f"{t:%M}", f"{t:%S}"]

    fields = [
        str(row_id), *parts(ts), *parts(te), f"{td:.3f}", "10.0.0.1", "10.0.0.2", str(sp), str(dp), pr, *flags,
        "0", "0", "2", "120", "0", "0", "616", "0", "2847", "2847",
        "0", "0", "0", "0", "0.0.0.0", "0.0.0.0", "0", "0", *["00:00:00:00:00:00"] * 4, *["0-0-0"] * 10,
        *["    0.000"] * 3, "0.0.0.0", "0/0", "5", "1970-01-01 03:00:00.000", *rules, attack_t, str(attack_a),
    ]
    assert len(fields) == 85
    return ",".join(f'"{f}"' for f in fields)


def write_archive(path, member, lines):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, "".join(line + "\r\n" for line in lines))


def capture_rows(start, kinds, seed):
    # one row per kind, end times spread over a minute with deliberate ties, written in shuffled order
    rng = random.Random(seed)
    offsets = [rng.randrange(0, 60) // 4 * 4 for _ in kinds]
    order = list(range(len(kinds)))
    rng.shuffle(order)
    rows = []
    for row_id, i in enumerate(order, start=1):
        attack_t, pr, flags = kinds[i]
        rows.append({"row_id": row_id, "te": start + timedelta(seconds=offsets[i]), "attack_t": attack_t,
                     "attack_a": int(attack_t != "none"), "pr": pr, "flags": flags})
    return rows


def build(tmp_path, monkeypatch, start_a, start_b):
    storage = tmp_path / "storage"
    monkeypatch.setenv("FORGE_STORAGE", str(storage))
    cfg = load_config(configs_dir() / "data" / "litnet2020.yaml")
    raw = storage / "data" / "raw" / cfg["dataset"]["raw_dir"]
    raw.mkdir(parents=True)
    # capture a mixes two families on icmp, capture b mixes benign tcp with udp flood flows and hex flag strings
    a = capture_rows(start_a, [("none", "ICMP", "......")] * 9 + [("icmp_smf", "ICMP", "......")] * 4
                     + [("icmp_f", "ICMP", "......")] * 2, seed=1)
    b = capture_rows(start_b, [("none", "TCP", "....S.")] * 10 + [("none", "TCP", "  0xc2")] * 2
                     + [("udp_f", "UDP", "......")] * 5, seed=2)
    for name, member, rows in (("A_FLOWS.zip", "A_v2.csv", a), ("B_FLOWS.zip", "B_v2.csv", b),
                               ("C_FLOWS.zip", "C_v2.csv", a)):
        lines = [litnet_line(r["row_id"], r["te"], r["attack_t"], r["attack_a"], pr=r["pr"], flags=r["flags"])
                 for r in rows]
        write_archive(raw / name, member, lines)
    files = {}
    for name in ("A_FLOWS.zip", "B_FLOWS.zip", "C_FLOWS.zip"):
        data = (raw / name).read_bytes()
        files[name] = (len(data), hashlib.sha256(data).hexdigest())
    rows = [{**r, "capture": "a"} for r in a] + [{**r, "capture": "b"} for r in b]
    return SimpleNamespace(cfg=cfg, raw=raw, files=files, duplicates={"C_FLOWS.zip": "A_FLOWS.zip"}, rows=rows)


@pytest.fixture
def litnet_raw(tmp_path, monkeypatch):
    # b in the week of 2019-12-30, a in the week of 2020-01-13, the week between them holds nothing
    return build(tmp_path, monkeypatch, start_a=datetime(2020, 1, 15, 10, 0, 0), start_b=datetime(2020, 1, 1, 10, 0, 0))


@pytest.fixture
def litnet_overlapping(tmp_path, monkeypatch):
    return build(tmp_path, monkeypatch, start_a=datetime(2020, 1, 1, 10, 0, 20), start_b=datetime(2020, 1, 1, 10, 0, 0))
