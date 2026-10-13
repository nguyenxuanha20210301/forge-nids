"""tiny archives in the layout of the published LITNET-2020 and AnoShift files, and a small processed AnoShift table
whose features carry the label, written under a temporary storage root."""

import hashlib
import json
import random
import tarfile
import zipfile
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forge.config import load_config
from forge.data import anoshift
from forge.data.litnet2020 import capture_name
from forge.paths import configs_dir

# archive, first flow end and attack codes of each published capture, the icmp flood archive holds two families
PUBLISHED = [
    ("UDP_FLOOD_FLOWS.zip", datetime(2019, 3, 6, 21, 13, 58), ["udp_f"]),
    ("REAPER_WORM_FLOWS.zip", datetime(2019, 10, 2, 7, 48, 53), ["udp_reaper_w"]),
    ("FRAGMENTATION_FLOWS.zip", datetime(2019, 11, 15, 5, 8, 59), ["udp_0"]),
    ("SYN_FLOOD_FLOWS.zip", datetime(2019, 12, 3, 2, 44, 56), ["tcp_syn_f"]),
    ("SPAM_FLOWS.zip", datetime(2020, 1, 6, 2, 13, 59), ["smtp_b"]),
    ("HTTP_FLOOD_FLOWS.zip", datetime(2020, 1, 16, 9, 28, 57), ["http_f"]),
    ("SCANNING_SPREAD_FLOWS.zip", datetime(2020, 1, 21, 8, 38, 58), ["tcp_udp_win_p"]),
    ("BLASTER_WORM_FLOWS.zip", datetime(2020, 1, 25, 3, 43, 56), ["tcp_w32_w"]),
    ("LAND_ATTACK_FLOWS.zip", datetime(2020, 1, 30, 12, 13, 56), ["tcp_land"]),
    ("RED_WORM_FLOWS.zip", datetime(2020, 1, 30, 20, 31, 7), ["tcp_red_w"]),
    ("ICMP_FLOOD_FLOWS.zip", datetime(2020, 1, 31, 15, 34, 58), ["icmp_smf", "icmp_f"]),
]


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


def write_raw(tmp_path, monkeypatch, captures, duplicates):
    # captures maps an archive name to its rows, duplicates maps a copy to the archive whose csv it repeats
    storage = tmp_path / "storage"
    monkeypatch.setenv("FORGE_STORAGE", str(storage))
    cfg = load_config(configs_dir() / "data" / "litnet2020.yaml")
    raw = storage / "data" / "raw" / cfg["dataset"]["raw_dir"]
    raw.mkdir(parents=True)
    archives = {**captures, **{copy: captures[original] for copy, original in duplicates.items()}}
    for name, rows in archives.items():
        lines = [litnet_line(r["row_id"], r["te"], r["attack_t"], r["attack_a"], pr=r["pr"], flags=r["flags"])
                 for r in rows]
        write_archive(raw / name, name.replace("_FLOWS.zip", "_v2.csv"), lines)
    files = {}
    for name in sorted(archives):
        data = (raw / name).read_bytes()
        files[name] = (len(data), hashlib.sha256(data).hexdigest())
    rows = [{**r, "capture": capture_name(name)} for name, capture in captures.items() for r in capture]
    return SimpleNamespace(cfg=cfg, raw=raw, files=files, duplicates=duplicates, rows=rows)


def two_captures(tmp_path, monkeypatch, start_a, start_b):
    # capture a mixes two families on icmp, capture b mixes benign tcp with udp flood flows and hex flag strings
    a = capture_rows(start_a, [("none", "ICMP", "......")] * 9 + [("icmp_smf", "ICMP", "......")] * 4
                     + [("icmp_f", "ICMP", "......")] * 2, seed=1)
    b = capture_rows(start_b, [("none", "TCP", "....S.")] * 10 + [("none", "TCP", "  0xc2")] * 2
                     + [("udp_f", "UDP", "......")] * 5, seed=2)
    return write_raw(tmp_path, monkeypatch, {"A_FLOWS.zip": a, "B_FLOWS.zip": b}, {"C_FLOWS.zip": "A_FLOWS.zip"})


@pytest.fixture
def litnet_raw(tmp_path, monkeypatch):
    # b in the week of 2019-12-30, a in the week of 2020-01-13, the week between them holds nothing
    return two_captures(tmp_path, monkeypatch, start_a=datetime(2020, 1, 15, 10, 0, 0),
                        start_b=datetime(2020, 1, 1, 10, 0, 0))


@pytest.fixture
def litnet_overlapping(tmp_path, monkeypatch):
    return two_captures(tmp_path, monkeypatch, start_a=datetime(2020, 1, 1, 10, 0, 20),
                        start_b=datetime(2020, 1, 1, 10, 0, 0))


@pytest.fixture
def litnet_like(tmp_path, monkeypatch):
    # the eleven captures and the duplicate smurf archive of the published dataset, a few flows each, on their dates
    captures = {
        name: capture_rows(start, [("none", "TCP", "....S.")] * 6 + [(code, "TCP", "....S.") for code in codes] * 2,
                           seed=i)
        for i, (name, start, codes) in enumerate(PUBLISHED)
    }
    return write_raw(tmp_path, monkeypatch, captures, {"SMURF_FLOWS.zip": "ICMP_FLOOD_FLOWS.zip"})


# label counts known, unknown and normal of the tiny AnoShift test files, one year from each period of the protocol
ANOSHIFT_YEARS = {2006: (40, 5, 15), 2011: (60, 4, 16), 2014: (55, 1, 14)}
ANOSHIFT_SERVICES = ["other", "ssh", "smtp", "dns", "http", "smtp,ssl"]
ANOSHIFT_FLAGS = ["S0", "SF", "OTH", "REJ", "RSTO"]


def anoshift_frame(counts, seed):
    # one test file in the published layout, string columns, rows in shuffled order and the row position in the year
    # as the pandas index, a sample of positions as in a 10% file
    rng = np.random.default_rng(seed)
    codes = ["-1"] * counts[0] + ["-2"] * counts[1] + ["1"] * counts[2]
    n = len(codes)
    rng.shuffle(codes)

    def rates():
        return [f"{v:.2f}" for v in rng.integers(0, 101, n) / 100]

    frame = pd.DataFrame({
        "0": [f"c0{b}" for b in rng.integers(0, 232, n)],
        "1": rng.choice(ANOSHIFT_SERVICES, n),
        "2": [f"c2{b}" for b in rng.integers(0, 232, n)],
        "3": [f"c3{b}" for b in rng.integers(0, 232, n)],
        "4": rng.integers(0, 101, n).astype(str),
        "5": rates(), "6": rates(), "7": rates(),
        "8": rng.integers(0, 101, n).astype(str),
        "9": rng.integers(0, 101, n).astype(str),
        "10": rates(), "11": rates(), "12": rates(),
        "13": rng.choice(ANOSHIFT_FLAGS, n),
        "18": codes,
    }, index=pd.Index(rng.choice(10 * n, size=n, replace=False)))
    # a value above the last bin edge, written nan by the authors
    frame.iloc[0, frame.columns.get_loc("2")] = "nan"
    return frame


def write_anoshift_tar(folder, years):
    # the tar layout of the published data, the test files among train and subset files that download must skip
    root = folder / "Kyoto-2016_AnoShift"
    (root / "full").mkdir(parents=True)
    (root / "subset").mkdir()
    frames, expected = {}, {}
    for i, (year, counts) in enumerate(years.items()):
        frames[year] = anoshift_frame(counts, seed=10 + i)
        test = root / "full" / f"{year}_full_valid.parquet"
        frames[year].to_parquet(test)
        anoshift_frame(counts, seed=20 + i).to_parquet(root / "full" / f"{year}_full.parquet")
        anoshift_frame(counts, seed=30 + i).to_parquet(root / "subset" / f"{year}_subset.parquet")
        expected[year] = {"bytes": test.stat().st_size, "rows": sum(counts), "known": counts[0], "unknown": counts[1],
                          "normal": counts[2]}
    tar = folder / "Kyoto-2016_AnoShift.tar"
    with tarfile.open(tar, "w") as out:
        out.add(root, arcname=root.name)
    return tar, expected, frames


@pytest.fixture
def anoshift_source(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("FORGE_STORAGE", str(storage))
    cfg = load_config(configs_dir() / "data" / "anoshift.yaml")
    tar, expected, frames = write_anoshift_tar(tmp_path / "source", ANOSHIFT_YEARS)
    return SimpleNamespace(cfg=cfg, tar=tar, url=tar.as_uri(), tar_bytes=tar.stat().st_size,
                           tar_sha256=hashlib.sha256(tar.read_bytes()).hexdigest(), expected=expected, frames=frames)


# label counts known, unknown and normal per year of a processed AnoShift table whose features carry the label, six
# years so that train, selection, calibration and deployment each get their own. the last year has no unknown type,
# as some blocks of the real FAR years have none
ANOSHIFT_SIGNAL_YEARS = {2006: (1500, 40, 1500), 2007: (1200, 30, 1800), 2008: (1600, 40, 2400),
                         2009: (3000, 60, 4000), 2010: (2400, 50, 1600), 2011: (2600, 0, 1400)}
# the same with nearly as many unknown as known attacks in 2009, so that the unknown type can be its own family
ANOSHIFT_FAMILY_YEARS = {**ANOSHIFT_SIGNAL_YEARS, 2009: (2000, 1500, 4000)}


def anoshift_signal_frame(years, seed):
    # attacks have more SYN errors and use other services, the unknown type sits between benign sessions and known
    # attacks, so the low tail of the attack scores holds misses of both types. the other columns are noise. the last
    # year brings a service that the earlier years never show
    rng = np.random.default_rng(seed)
    frames = []
    for i, (year, (known, unknown, normal)) in enumerate(years.items()):
        n = known + unknown + normal
        kind = np.array(["known"] * known + ["unknown"] * unknown + ["benign"] * normal)
        rng.shuffle(kind)
        attack = kind != "benign"
        level = np.select([kind == "known", kind == "unknown"], [0.62, 0.45], 0.3)

        def rate(signal):
            values = rng.normal(level, 0.15) if signal else rng.random(n)
            return np.clip(values, 0, 1).round(2).astype(np.float32)

        services = ANOSHIFT_SERVICES + (["ftp"] if i == len(years) - 1 else [])
        service = np.where(attack & (rng.random(n) < 0.4), "other", rng.choice(services, n))
        frames.append(pd.DataFrame({
            "year": np.full(n, year, dtype=np.int16),
            "order": np.sort(rng.choice(10 * n, size=n, replace=False)).astype(np.int32),
            "label": attack.astype(np.int8),
            "attack_type": pd.Categorical(kind, categories=anoshift.ATTACK_TYPES),
            "duration_bin": rng.integers(-1, 60, n).astype(np.int16),
            "service": pd.Categorical(service),
            "src_bytes_bin": rng.integers(-1, 120, n).astype(np.int16),
            "dst_bytes_bin": rng.integers(0, 120, n).astype(np.int16),
            "count": rng.integers(0, 101, n).astype(np.int16),
            "same_srv_rate": rate(False),
            "serror_rate": rate(True),
            "srv_serror_rate": rate(False),
            "dst_host_count": rng.integers(0, 101, n).astype(np.int16),
            "dst_host_srv_count": rng.integers(0, 101, n).astype(np.int16),
            "dst_host_same_src_port_rate": rate(False),
            "dst_host_serror_rate": rate(True),
            "dst_host_srv_serror_rate": rate(False),
            "flag": pd.Categorical(rng.choice(ANOSHIFT_FLAGS, n)),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def anoshift_signal(tmp_path, monkeypatch, request):
    # written where anoshift.load finds it, in small row groups so that reading some years skips the others, with the
    # one manifest field the experiments read. a test may pass other counts per year through indirect parametrization
    monkeypatch.setenv("FORGE_STORAGE", str(tmp_path / "storage"))
    cfg = load_config(configs_dir() / "data" / "anoshift.yaml")
    frame = anoshift_signal_frame(getattr(request, "param", ANOSHIFT_SIGNAL_YEARS), seed=5)
    out = anoshift.processed_dir(cfg)
    out.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(frame, schema=anoshift.SCHEMA, preserve_index=False), out / "flows.parquet",
                   row_group_size=1000)
    (out / "manifest.json").write_text(json.dumps({"fingerprint": anoshift.fingerprint(frame)}), encoding="utf-8")
    return SimpleNamespace(cfg=cfg, frame=frame)


# captures of a processed LITNET-2020 table whose features carry the label, in time order, with the benign flows and
# the attacks of each family and the seconds of the six minute capture in which each family attacks. udp flood holds
# udp benign flows only, as in the published capture, and the two icmp families follow each other without overlap
LITNET_SIGNAL_CAPTURES = [
    ("udp_flood", datetime(2019, 3, 6, 21, 13, 58), 400, {"udp_flood": (1000, 0, 360)}),
    ("reaper_worm", datetime(2019, 10, 2, 7, 48, 53), 7600, {"reaper_worm": (80, 20, 200)}),
    ("syn_flood", datetime(2019, 12, 3, 2, 44, 56), 7400, {"syn_flood": (1600, 120, 360)}),
    ("spam", datetime(2020, 1, 6, 2, 13, 59), 7400, {"spam_bot": (70, 0, 360)}),
    ("icmp_flood", datetime(2020, 1, 31, 15, 34, 58), 7800, {"smurf": (700, 0, 150), "icmp_flood": (450, 200, 360)}),
    ("red_worm", datetime(2020, 1, 30, 20, 31, 7), 7500, {"code_red_worm": (1300, 0, 360)}),
]
# how far each family's attacks sit from the benign flows in the features that carry the label, the pooled families
# closest, so the low tail of the attack scores holds misses
LITNET_SIGNAL_LEVELS = {"udp_flood": 0.62, "reaper_worm": 0.53, "syn_flood": 0.66, "spam_bot": 0.55, "smurf": 0.6,
                        "icmp_flood": 0.56, "code_red_worm": 0.6}
LITNET_PROTOCOLS = {"udp_flood": "UDP", "reaper_worm": "UDP", "syn_flood": "TCP", "spam_bot": "TCP", "smurf": "ICMP",
                    "icmp_flood": "ICMP", "code_red_worm": "TCP"}


def litnet_signal_frame(captures, seed):
    # the processed v1 layout, rows of a capture sorted by te and row_id, row ids in a shuffled source order. the
    # benign flows of each capture sit a little apart from those of the others, as captures of other days would
    from forge.data import litnet2020

    rng = np.random.default_rng(seed)
    names = [name for name, _, _, _ in sorted(captures, key=lambda c: c[1])]
    frames = []
    for i, (name, start, benign, families) in enumerate(sorted(captures, key=lambda c: c[1])):
        kind = ["benign"] * benign + [f for f, (n, _, _) in families.items() for _ in range(n)]
        # benign flows over the whole capture, six minutes or the last second of an attack
        last = max([360] + [hi for _, _, hi in families.values()])
        second = np.r_[rng.integers(0, last + 1, benign),
                       np.concatenate([rng.integers(lo, hi + 1, n) for n, lo, hi in families.values()])]
        n = len(kind)
        kind = np.array(kind)
        attack = kind != "benign"
        level = np.where(attack, [LITNET_SIGNAL_LEVELS.get(k, 0.0) for k in kind], 0.3 + 0.02 * (i % 3))

        def signal():
            # one noisy look at the level of every flow, each feature that carries the label takes its own
            return np.clip(rng.normal(level, 0.15), 0, 1)

        proto = np.where(attack, [LITNET_PROTOCOLS.get(k, "TCP") for k in kind],
                         rng.choice(["TCP", "UDP", "ICMP"], n, p=[0.7, 0.25, 0.05]))
        if name == "udp_flood":
            proto[~attack] = "UDP"
        te = (np.datetime64(start, "ms") + second.astype("timedelta64[s]")).astype("datetime64[ms]")
        td = (10 * signal()).round(3)
        frame = pd.DataFrame({
            "ts": te - (td * 1000).astype("timedelta64[ms]"),
            "te": te,
            "label": attack.astype(np.int8),
            "family": pd.Categorical(kind, categories=litnet2020.FAMILY_ORDER),
            "capture": pd.Categorical([name] * n, categories=names),
            "row_id": rng.permutation(n).astype(np.int32) + 1,
            "td": td,
            "proto": pd.Categorical(proto, categories=["ICMP", "TCP", "UDP"]),
            "tcp_flags": np.where(attack & (rng.random(n) < 0.5), 2, rng.choice([2, 16, 18, 24, 27], n)).astype(
                np.uint8),
            "stos": rng.choice([0, 0, 40], n).astype(np.uint8),
            "ipkt": np.maximum(1, np.round(np.exp(4 * signal()))).astype(np.int64),
            "ibyt": np.maximum(40, np.round(np.exp(8 * signal()))).astype(np.int64),
            "sp": np.where(rng.random(n) < 0.1, rng.integers(1, 1024, n), rng.integers(1024, 65536, n)).astype(
                np.int32),
            "dp": rng.choice([0, 25, 53, 80, 443, 2048], n).astype(np.int32),
            "in_if": rng.integers(1, 5, n).astype(np.int32),
            "out_if": rng.integers(1, 5, n).astype(np.int32),
            "sas": rng.choice([0, 8764], n).astype(np.int64),
            "das": rng.choice([0, 8764], n).astype(np.int64),
            "exid": rng.choice([1, 2], n).astype(np.int16),
        })
        frames.append(frame.sort_values(["te", "row_id"], kind="stable", ignore_index=True))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def litnet_signal(tmp_path, monkeypatch, request):
    # written where litnet2020.load finds it, with the manifest fields the experiments read. a test may pass other
    # captures through indirect parametrization
    from forge.data import litnet2020

    monkeypatch.setenv("FORGE_STORAGE", str(tmp_path / "storage"))
    cfg = load_config(configs_dir() / "data" / "litnet2020.yaml")
    frame = litnet_signal_frame(getattr(request, "param", LITNET_SIGNAL_CAPTURES), seed=7)
    out = litnet2020.processed_dir(cfg)
    out.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(frame, schema=litnet2020.SCHEMA, preserve_index=False), out / "flows.parquet",
                   row_group_size=5000)
    (out / "manifest.json").write_text(json.dumps({"time_column": "te", "fingerprint": litnet2020.fingerprint(frame)}),
                                       encoding="utf-8")
    return SimpleNamespace(cfg=cfg, frame=frame)
