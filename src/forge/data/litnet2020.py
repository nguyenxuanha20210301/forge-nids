"""LITNET-2020 as published by its authors, downloaded, checked and written as one time ordered table.

the authors publish the dataset in their GitHub repository, where the address given in the paper, dataset.litnet.lt,
now redirects. it is twelve zip files, one per attack type, each holding one csv without a header row and 85 quoted
columns per flow. each csv is a capture of about six minutes on one day between 2019-03-06 and 2020-01-31, benign
traffic together with the flows of one attack, so the data is about 73 minutes of traffic in short captures and not a
continuous stream. the smurf and icmp flood archives hold the same csv, which carries both attack types, so eleven
captures remain.

the syn flood archive holds only the first 6012524 rows of its capture, which end at 2019-12-03 02:56:37 while the
attack went on until 03:10:33. it carries 1580016 of the 3725838 syn flood attacks in the paper. the other 2145822 are
in SYN_FLOOD_ATTACKERS_ONLY.zip alone, whose row numbers count from the start of the original capture, without the
benign flows of the same minutes, so adding them would raise the prevalence of the capture and they are left out.

download fetches the archives at a fixed commit and keeps a file only when its size and sha256 match the published
ones. prepare checks the hashes again, reads the eleven captures and writes the flow times, the label, the attack
family and the flow features into one parquet file sorted by time, inside a versioned folder with a readme and a
manifest that record the source, the hashes, the counts and a content fingerprint. capture_parts cuts each capture of
that table in time order into the parts of an experiment, at the ranks of its attacks.
"""

import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from forge import paths
from forge.data import digest, splits
from forge.runs import git_state

PAPER = (
    "Damasevicius R., Venckauskas A., Grigaliunas S., Toldinas J., Morkevicius N., Aleliunas T., Smuikys P. "
    "LITNET-2020: An Annotated Real-World Network Flow Dataset for Network Intrusion Detection. "
    "Electronics 9(5):800, 2020. doi:10.3390/electronics9050800."
)
REPO = "Grigaliunas/electronics9050800"
COMMIT = "7b366c83d3c255d0507f706c7796af4f21a37bc8"
# github.com/<repo>/raw/<commit>/<path> redirects to raw.githubusercontent.com for plain files and to
# media.githubusercontent.com for git lfs files, so one pattern serves all twelve archives
URL = f"https://github.com/{REPO}/raw/{COMMIT}/dataset/{{name}}"

# size in bytes and sha256 of every capture archive at COMMIT. the sha256 of the seven git lfs files comes from their
# lfs pointers, the other five were hashed after their git blob sha1 matched the one GitHub reports
FILES = {
    "BLASTER_WORM_FLOWS.zip": (74424976, "ea2f9093b06c98479e31bbc7602ea6bf29d5826aa83608e23a0f8e62c5b25837"),
    "FRAGMENTATION_FLOWS.zip": (45814412, "c92807741c45c7d3261651802fe88f10cc2b6720ea1eab8fb35e7d88874bf390"),
    "HTTP_FLOOD_FLOWS.zip": (146438614, "42d495e96f9602b749f3f61c78533a37e41eeefe625199dfae31b4a11567d865"),
    "ICMP_FLOOD_FLOWS.zip": (129244901, "5195332963af9c227bf85f8fb93af206733b61e35eaf68dc21421a3131638bd1"),
    "LAND_ATTACK_FLOWS.zip": (144452440, "0f8fe47aa1d9cbb8e5a99bd1dc666a3b5ddd282cb59b13a07e6d211b084cd1e8"),
    "REAPER_WORM_FLOWS.zip": (162850983, "8504413ef0fb5382ebd4f5038497de7df11c18d457d9d3c2ada2968b6bb0b380"),
    "RED_WORM_FLOWS.zip": (161155999, "2d2f2542351a1141e1b8a0b476b47f68bcec5ed622b29d839319014bf19297fc"),
    "SCANNING_SPREAD_FLOWS.zip": (104591, "1817eb649d524c14c45b2ae065ec9aaac76d55d34143aac85c6e3b683c6ed7bb"),
    "SMURF_FLOWS.zip": (129244891, "0561ec80121cce1e3adac857782ca6868fbb5fc67fd21db45d98c285edfea50b"),
    "SPAM_FLOWS.zip": (42726018, "a1d50e0dc8fc7ee06012dbc6f5a05d99e143d9485084f2a13a05c15bd7a18c9c"),
    "SYN_FLOOD_FLOWS.zip": (175517521, "7d6764d7278f838b4cc870ac35bed8a29e009451aba7d26b7f7f4dabc2f5748c"),
    "UDP_FLOOD_FLOWS.zip": (17491937, "dd30f17f9c19fa3126e30ec6f84a9389102b1bd75b928798bc89a735b16939e6"),
}
# the smurf archive holds the same csv as the icmp flood archive, prepare checks this again before skipping it
DUPLICATES = {"SMURF_FLOWS.zip": "ICMP_FLOOD_FLOWS.zip"}

# what each processed version changed, written into its readme
VERSIONS = {"v1": "First version."}

# the per attack csv files have no header, these names are the first line of allFlows.csv inside ALLinONE.zip
COLUMNS = (
    "ID",
    "ts_year", "ts_month", "ts_day", "ts_hour", "ts_min", "ts_second",
    "te_year", "te_month", "te_day", "te_hour", "te_min", "te_second",
    "td", "sa", "da", "sp", "dp", "pr",
    "_flag1", "_flag2", "_flag3", "_flag4", "_flag5", "_flag6",
    "fwd", "stos", "ipkt", "ibyt", "opkt", "obyt", "_in", "out", "sas", "das",
    "smk", "dmk", "dtos", "_dir", "nh", "nhb", "svln", "dvln", "ismc", "odmc", "idmc", "osmc",
    "mpls1", "mpls2", "mpls3", "mpls4", "mpls5", "mpls6", "mpls7", "mpls8", "mpls9", "mpls10",
    "cl", "sl", "al", "ra", "eng", "exid", "tr",
    "icmp_dst_ip_b", "icmp_src_ip", "udp_dst_p", "tcp_f_s", "tcp_f_n_a", "tcp_f_n_f", "tcp_f_n_r", "tcp_f_n_p",
    "tcp_f_n_u", "tcp_dst_p", "tcp_src_dst_f_s", "tcp_src_tftp", "tcp_src_kerb", "tcp_src_rpc", "tcp_dst_p_src",
    "smtp_dst", "udp_p_r_range", "p_range_dst", "udp_src_p_0",
    "attack_t", "attack_a",
)
# outputs of the labelling rules, blank on attack rows except for the rule that fired and normal on benign rows, so
# they repeat the label and never reach a model
RULE_COLUMNS = COLUMNS[COLUMNS.index("icmp_dst_ip_b"):COLUMNS.index("udp_src_p_0") + 1]
# one value in every file read while writing this module, the manifest records the values each version actually saw
CONSTANT_COLUMNS = (
    "fwd", "opkt", "obyt", "smk", "dmk", "dtos", "_dir", "nh", "nhb", "svln", "dvln", "ismc", "odmc", "idmc", "osmc",
    "mpls1", "mpls2", "mpls3", "mpls4", "mpls5", "mpls6", "mpls7", "mpls8", "mpls9", "mpls10",
    "cl", "sl", "al", "ra", "eng", "tr",
)
# addresses identify hosts and look unanonymised, so they stay in the raw files only
IDENTIFIER_COLUMNS = ("sa", "da")
FLAG_COLUMNS = ("_flag1", "_flag2", "_flag3", "_flag4", "_flag5", "_flag6")
TIME_UNITS = ("year", "month", "day", "hour", "min", "second")
TIME_KEYS = ("year", "month", "day", "hour", "minute", "second")

USECOLS = [c for c in COLUMNS if c not in RULE_COLUMNS and c not in IDENTIFIER_COLUMNS]
# integers are read wide and narrowed only after a range check, categories avoid one python string per cell
DTYPES = {
    "ID": "int64",
    **{f"{p}_{u}": "int16" for p in ("ts", "te") for u in TIME_UNITS},
    "td": "float64",
    **{c: "int64" for c in ("sp", "dp", "stos", "ipkt", "ibyt", "_in", "out", "sas", "das", "exid", "attack_a")},
    **{c: "category" for c in ("pr", "attack_t", *FLAG_COLUMNS, *CONSTANT_COLUMNS)},
}

# attack_t codes from Table 6 of the paper, in its order, and the family name used here
FAMILIES = {
    "none": "benign",
    "icmp_smf": "smurf",
    "icmp_f": "icmp_flood",
    "udp_f": "udp_flood",
    "tcp_syn_f": "syn_flood",
    "http_f": "http_flood",
    "tcp_land": "land",
    "tcp_w32_w": "blaster_worm",
    "tcp_red_w": "code_red_worm",
    "smtp_b": "spam_bot",
    "udp_reaper_w": "reaper_worm",
    "tcp_udp_win_p": "scanning",
    "udp_0": "fragmentation",
}
FAMILY_ORDER = list(FAMILIES.values())
ATTACK_FAMILIES = FAMILY_ORDER[1:]
# family names as figures show them
FAMILY_LABELS = {
    "benign": "benign",
    "smurf": "Smurf",
    "icmp_flood": "ICMP flood",
    "udp_flood": "UDP flood",
    "syn_flood": "SYN flood",
    "http_flood": "HTTP flood",
    "land": "LAND",
    "blaster_worm": "Blaster worm",
    "code_red_worm": "Code Red worm",
    "spam_bot": "spam bot",
    "reaper_worm": "Reaper worm",
    "scanning": "scanning",
    "fragmentation": "fragmentation",
}
# flows and attacks per attack type in Table 4 of the paper, compared with the published files in the readme. the
# attacks of the rows sum to 5254529, while the total row of the table says 5328934
PAPER_TABLE4 = {
    "smurf": (3994426, 59479),
    "icmp_flood": (3863655, 11628),
    "udp_flood": (606814, 93583),
    "syn_flood": (14608678, 3725838),
    "http_flood": (3963168, 22959),
    "land": (3569838, 52417),
    "blaster_worm": (2858573, 24291),
    "code_red_worm": (5082952, 1255702),
    "spam_bot": (1153020, 747),
    "reaper_worm": (4377656, 1176),
    "scanning": (6687, 6232),
    "fragmentation": (1244866, 477),
}

# Parquet has no second unit, so times are stored in ms and keep their whole second values
SCHEMA = pa.schema([
    ("ts", pa.timestamp("ms")),
    ("te", pa.timestamp("ms")),
    ("label", pa.int8()),
    ("family", pa.dictionary(pa.int8(), pa.string())),
    ("capture", pa.dictionary(pa.int8(), pa.string())),
    ("row_id", pa.int32()),
    ("td", pa.float64()),
    ("proto", pa.dictionary(pa.int16(), pa.string())),
    ("tcp_flags", pa.uint8()),
    ("stos", pa.uint8()),
    ("ipkt", pa.int64()),
    ("ibyt", pa.int64()),
    ("sp", pa.int32()),
    ("dp", pa.int32()),
    ("in_if", pa.int32()),
    ("out_if", pa.int32()),
    ("sas", pa.int64()),
    ("das", pa.int64()),
    ("exid", pa.int16()),
])
# source and meaning of each column of the processed table, written into the readme
COLUMN_NOTES = {
    "ts": ("ts_year to ts_second", "time", "flow start, as written, 1 s resolution"),
    "te": ("te_year to te_second", "time", "flow end, as written, 1 s resolution"),
    "label": ("attack_a", "label", "1 attack, 0 benign"),
    "family": ("attack_t", "label", "attack family, benign for benign flows"),
    "capture": ("archive name", "provenance", "capture the flow comes from"),
    "row_id": ("ID", "provenance", "line of the flow in the source CSV, counted from 1"),
    "td": ("td", "feature", "duration in seconds"),
    "proto": ("pr", "feature", "protocol name as written by nfdump"),
    "tcp_flags": ("_flag1 to _flag6", "feature", "TCP flag bitmask, CWR 0x80 down to FIN 0x01"),
    "stos": ("stos", "feature", "source type of service byte"),
    "ipkt": ("ipkt", "feature", "packets"),
    "ibyt": ("ibyt", "feature", "bytes"),
    "sp": ("sp", "identifier-like feature", "source port"),
    "dp": ("dp", "feature", "destination port, ICMP type times 256 plus code for ICMP"),
    "in_if": ("_in", "identifier-like feature", "SNMP index of the input interface"),
    "out_if": ("out", "identifier-like feature", "SNMP index of the output interface"),
    "sas": ("sas", "identifier-like feature", "source AS number"),
    "das": ("das", "identifier-like feature", "destination AS number"),
    "exid": ("exid", "identifier-like feature", "exporter id"),
}

CHUNK_ROWS = 1_000_000
ROW_GROUP_ROWS = 1_000_000


def capture_name(file_name: str) -> str:
    return file_name.removesuffix("_FLOWS.zip").lower()


def raw_dir(cfg: dict) -> Path:
    return paths.data_dir("raw") / cfg["dataset"]["raw_dir"]


def processed_dir(cfg: dict) -> Path:
    d = cfg["dataset"]
    return paths.data_dir("processed") / d["name"] / d["processed_version"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def download(cfg: dict, log=print, files: dict | None = None, url: str = URL) -> Path:
    # a file gets its final name only after its size and sha256 match, so a name in the raw folder means a checked copy
    files = FILES if files is None else files
    folder = raw_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in files.items():
        path = folder / name
        if path.exists() and path.stat().st_size == size:
            log(f"found {name}, {size} bytes")
            continue
        part = folder / f"{name}.part"
        h = hashlib.sha256()
        with urllib.request.urlopen(url.format(name=name), timeout=120) as response, open(part, "wb") as f:
            while block := response.read(1 << 20):
                f.write(block)
                h.update(block)
        if part.stat().st_size != size or h.hexdigest() != digest:
            part.unlink()
            raise ValueError(f"{name} arrived with a size or sha256 other than the published one and was removed")
        part.replace(path)
        log(f"downloaded {name}, {size} bytes, sha256 matches")
    return folder


def verify_raw(folder: Path, files: dict, log=print) -> None:
    for name, (size, digest) in files.items():
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing, download fetches it")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"{path} differs from the published size or sha256")
    log(f"checked size and sha256 of {len(files)} raw files")


def _member(path: Path) -> zipfile.ZipInfo:
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
    if len(members) != 1:
        raise ValueError(f"{path.name} holds {len(members)} files where one csv was expected")
    return members[0]


def check_duplicates(folder: Path, duplicates: dict) -> None:
    # size and crc32 of the csv as stored in each archive, the archives themselves differ only in the member name
    for copy, original in duplicates.items():
        a, b = _member(folder / copy), _member(folder / original)
        if (a.file_size, a.CRC) != (b.file_size, b.CRC):
            raise ValueError(f"{copy} and {original} hold different csv files, so {copy} cannot be skipped")


def parse_times(raw: pd.DataFrame, prefix: str) -> np.ndarray:
    # each time is split into six integer columns, kept as written without a time zone. the unused tr column shows
    # 1970-01-01 03:00, epoch zero printed in a UTC+3 zone, so the collector most likely wrote local Lithuanian time
    parts = pd.DataFrame({key: raw[f"{prefix}_{unit}"].to_numpy() for unit, key in zip(TIME_UNITS, TIME_KEYS)})
    return pd.to_datetime(parts).to_numpy().astype("datetime64[s]")


def flag_value(text: str) -> int:
    # nfdump 1.6 writes the six letters UAPRSF with a dot for each unset bit, or 0x and two hex digits right aligned
    # in six characters when a bit above 0x3f is set, and the dataset splits that string over six columns
    if len(text) == 6 and text.startswith("  0x") and all(ch in "0123456789abcdef" for ch in text[4:]):
        return int(text[4:], 16)
    if len(text) == 6 and all(ch in (".", letter) for ch, letter in zip(text, "UAPRSF")):
        return sum(bit for ch, bit in zip(text, (32, 16, 8, 4, 2, 1)) if ch != ".")
    raise ValueError(f"unexpected tcp flags {text.strip()}")


def parse_flags(raw: pd.DataFrame) -> np.ndarray:
    # a few hundred distinct strings at most, so each is parsed once and spread over its rows
    text = raw[FLAG_COLUMNS[0]].astype(str)
    for c in FLAG_COLUMNS[1:]:
        text = text + raw[c].astype(str)
    text = text.astype("category")
    values = np.array([flag_value(t) for t in text.cat.categories], dtype=np.uint8)
    return values[text.cat.codes.to_numpy()]


def map_labels(attack_t: pd.Series, attack_a: pd.Series) -> tuple[np.ndarray, pd.Categorical]:
    # the label is attack_a and the family comes from attack_t, the two must agree on every row
    codes = attack_t.astype(str).astype("category")
    unknown = sorted(set(codes.cat.categories) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown attack_t values {', '.join(unknown)}")
    label = attack_a.to_numpy()
    if not np.isin(label, (0, 1)).all():
        raise ValueError("attack_a holds values other than 0 and 1")
    family = pd.Categorical(codes.map(FAMILIES), categories=FAMILY_ORDER)
    disagree = int(((family != "benign") != (label == 1)).sum())
    if disagree:
        raise ValueError(f"{disagree} rows where attack_a disagrees with attack_t")
    return label.astype(np.int8), family


def _fit(values, dtype, name: str) -> np.ndarray:
    # narrows an integer column only when every value fits, so nothing wraps around silently
    values = np.asarray(values)
    info = np.iinfo(dtype)
    if values.size and (values.min() < info.min or values.max() > info.max):
        raise ValueError(f"{name} holds values outside {info.min} to {info.max}")
    return values.astype(dtype)


def clean(raw: pd.DataFrame, capture: str, captures: list[str]) -> pd.DataFrame:
    label, family = map_labels(raw["attack_t"], raw["attack_a"])
    return pd.DataFrame({
        "ts": parse_times(raw, "ts"),
        "te": parse_times(raw, "te"),
        "label": label,
        "family": family,
        "capture": pd.Categorical.from_codes(np.full(len(raw), captures.index(capture)), categories=captures),
        "row_id": _fit(raw["ID"], np.int32, "ID"),
        "td": raw["td"].to_numpy(np.float64),
        "proto": pd.Categorical(raw["pr"].astype(str)),
        "tcp_flags": parse_flags(raw),
        "stos": _fit(raw["stos"], np.uint8, "stos"),
        "ipkt": raw["ipkt"].to_numpy(np.int64),
        "ibyt": raw["ibyt"].to_numpy(np.int64),
        "sp": _fit(raw["sp"], np.int32, "sp"),
        "dp": _fit(raw["dp"], np.int32, "dp"),
        "in_if": _fit(raw["_in"], np.int32, "_in"),
        "out_if": _fit(raw["out"], np.int32, "out"),
        "sas": raw["sas"].to_numpy(np.int64),
        "das": raw["das"].to_numpy(np.int64),
        "exid": _fit(raw["exid"], np.int16, "exid"),
    })


def read_capture(path: Path, captures: list[str], time_column: str, chunksize: int = CHUNK_ROWS):
    # reads one archive in chunks, so memory holds the cleaned columns of one capture and never its raw text
    capture = capture_name(path.name)
    member = _member(path)
    parts, seen = [], {c: set() for c in CONSTANT_COLUMNS}
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        reader = pd.read_csv(f, header=None, names=list(COLUMNS), usecols=USECOLS, dtype=DTYPES, na_filter=False,
                             chunksize=chunksize)
        for raw in reader:
            for c in CONSTANT_COLUMNS:
                seen[c].update(map(str, raw[c].cat.categories))
            try:
                parts.append(clean(raw, capture, captures))
            except ValueError as e:
                raise ValueError(f"{path.name}, rows from ID {raw['ID'].iloc[0]}, {e}") from e
    protos = sorted(set().union(*(p["proto"].cat.categories for p in parts)))
    for p in parts:
        p["proto"] = p["proto"].cat.set_categories(protos)
    flows = pd.concat(parts, ignore_index=True)
    # ties in time keep the order of the source file, which row_id records
    return flows.sort_values([time_column, "row_id"], kind="stable", ignore_index=True), seen


def fingerprint(flows: pd.DataFrame) -> str:
    # recomputes the fingerprint of a full table read back with load, one digest per capture block in row order
    return digest.fingerprint(flows, "capture", SCHEMA)


def _summary(flows: pd.DataFrame, name: str, time_column: str) -> dict:
    attacks = flows["label"].to_numpy() == 1
    families = flows.loc[attacks, "family"].value_counts()
    return {
        "capture": capture_name(name),
        "file": name,
        "flows": int(len(flows)),
        "attacks": int(attacks.sum()),
        "first": flows[time_column].iloc[0],
        "last": flows[time_column].iloc[-1],
        "ts_min": flows["ts"].min(),
        "te_max": flows["te"].max(),
        "families": {str(k): int(v) for k, v in families.items() if v > 0},
        "fingerprint": digest.table_digest(flows, SCHEMA),
    }


def prepare(cfg: dict, log=print, files: dict | None = None, duplicates: dict | None = None,
            chunksize: int = CHUNK_ROWS) -> Path:
    files = FILES if files is None else files
    duplicates = DUPLICATES if duplicates is None else duplicates
    d = cfg["dataset"]
    version, t = d["processed_version"], d["time_column"]
    if version not in VERSIONS:
        raise ValueError(f"VERSIONS has no description of {version}, the readme of a version says what it changed")
    if t not in ("ts", "te"):
        raise ValueError(f"time_column is ts or te, not {t}")
    out = processed_dir(cfg)
    if out.exists():
        raise FileExistsError(f"{out} exists and a processed version is never overwritten, name a new version")
    source = raw_dir(cfg)
    verify_raw(source, files, log)
    check_duplicates(source, duplicates)
    names = [n for n in files if n not in duplicates]
    captures = [capture_name(n) for n in names]

    # a version folder appears under its final name only once complete, a partial one is left by an interrupted run
    partial = out.with_name(f"{out.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    summaries, constants = [], {c: set() for c in CONSTANT_COLUMNS}
    with tempfile.TemporaryDirectory() as scratch:
        for name in names:
            flows, seen = read_capture(source / name, captures, t, chunksize)
            for c in CONSTANT_COLUMNS:
                constants[c] |= seen[c]
            s = _summary(flows, name, t)
            s["part"] = Path(scratch) / f"{s['capture']}.parquet"
            pq.write_table(pa.Table.from_pandas(flows, schema=SCHEMA, preserve_index=False), s["part"])
            summaries.append(s)
            log(f"read {name}, {s['flows']} flows, {s['attacks']} attacks, {t} from {s['first']} to {s['last']}")
            del flows
        summaries.sort(key=lambda s: (s["first"], s["last"]))
        for a, b in zip(summaries, summaries[1:]):
            if b["first"] < a["last"]:
                raise ValueError(f"captures {a['capture']} and {b['capture']} overlap in {t}, "
                                 "so the table cannot be written capture after capture")
        with pq.ParquetWriter(partial / "flows.parquet", SCHEMA, compression="zstd") as writer:
            for s in summaries:
                writer.write_table(pq.read_table(s.pop("part")), row_group_size=ROW_GROUP_ROWS)

    for c, values in constants.items():
        if len(values) > 1:
            log(f"column {c} is not constant, it holds {len(values)} values and is dropped all the same")
    manifest = {
        "dataset": d["name"],
        "version": version,
        "changes": VERSIONS[version],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {**git_state(), "python": sys.version.split()[0], "pandas": pd.__version__, "pyarrow": pa.__version__},
        "source": {
            "paper": PAPER,
            "repository": f"https://github.com/{REPO}",
            "commit": COMMIT,
            "files": {
                n: {"bytes": size, "sha256": digest, "used": n not in duplicates,
                    **({"same_csv_as": duplicates[n]} if n in duplicates else {})}
                for n, (size, digest) in files.items()
            },
        },
        "time_column": t,
        "rows": sum(s["flows"] for s in summaries),
        "attacks": sum(s["attacks"] for s in summaries),
        "columns": {f.name: str(f.type) for f in SCHEMA},
        "dropped": {
            "rule_columns": list(RULE_COLUMNS),
            "identifier_columns": list(IDENTIFIER_COLUMNS),
            "constant_columns": {c: sorted(v) for c, v in constants.items()},
        },
        "captures": [{k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in s.items()} for s in summaries],
        "fingerprint": digest.combine((s["capture"], s["fingerprint"]) for s in summaries),
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (partial / "readme.md").write_text(readme(manifest), encoding="utf-8")
    partial.rename(out)
    log(f"wrote {manifest['rows']} flows to {out}, fingerprint {manifest['fingerprint']}")
    return out


def read_manifest(cfg: dict) -> dict:
    return json.loads((processed_dir(cfg) / "manifest.json").read_text(encoding="utf-8"))


def load(cfg: dict, columns: list[str] | None = None) -> pd.DataFrame:
    # the processed table in the order it was written, sorted by the time column its manifest names
    manifest = read_manifest(cfg)
    if manifest["time_column"] != cfg["dataset"]["time_column"]:
        raise ValueError(f"{processed_dir(cfg)} is sorted by {manifest['time_column']} but the config asks for "
                         f"{cfg['dataset']['time_column']}, which needs its own processed version")
    flows = pq.read_table(processed_dir(cfg) / "flows.parquet", columns=columns).to_pandas()
    if "family" in flows.columns:
        flows["family"] = flows["family"].cat.set_categories(FAMILY_ORDER)
    return flows


def port_range(ports) -> np.ndarray:
    # 0, the well known ports 1 to 1023, the registered ones 1024 to 49151 and the dynamic ones from 49152 on, the IANA
    # ranges, so a port keeps what it says about the protocol and loses the value that can mark one host or tool run
    ports = np.asarray(ports)
    return np.select([ports == 0, ports < 1024, ports < 49152], [0, 1, 2], 3).astype(np.int8)


def _blocks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # first and end row of each run of equal values, every value in one run only
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    if len(np.unique(values[starts])) != len(starts):
        raise ValueError("the flows of a capture are not consecutive")
    return starts, np.r_[starts[1:], len(values)]


def capture_segments(capture, label, family, time) -> np.ndarray:
    """the segment of every flow of a table written capture after capture in time order, numbered from 0 in that order.

    a capture is one segment, unless the attacks of its families follow each other in time without overlap. it is then
    cut at the first attack of each later family, so that a split inside a segment finds the attacks of every family in
    every part. the benign flows between two families go with the earlier one, and those of the second in which a
    family starts with that family.
    """
    capture, label, family, time = (np.asarray(v) for v in (capture, label, family, time))
    segment = np.empty(len(capture), dtype=np.int32)
    count = 0
    for s, e in zip(*_blocks(capture)):
        t = time[s:e]
        if (t[1:] < t[:-1]).any():
            raise ValueError(f"the flows of capture {capture[s]} are not in time order")
        attack = label[s:e] == 1
        spans = sorted((t[attack & (family[s:e] == f)].min(), t[attack & (family[s:e] == f)].max())
                       for f in np.unique(family[s:e][attack]))
        cuts = []
        if len(spans) > 1 and all(a[1] < b[0] for a, b in zip(spans, spans[1:])):
            cuts = [int(np.searchsorted(t, b[0], side="left")) for b in spans[1:]]
        bounds = [0, *cuts, e - s]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            segment[s + lo:s + hi] = count
            count += 1
    return segment


def capture_parts(capture, label, family, time, shares, min_share: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """the segment and the part of every flow, parts numbered 0, 1 and on in the order of shares.

    each segment of capture_segments is cut at the ranks of its attacks, so every part holds its share of the segment's
    attacks whatever their timing. a segment with several families, whose attacks overlap in time, stops the split when
    one family holds less than min_share of its share in some part, since its parts would then not stand for it.
    """
    capture, label, family, time = (np.asarray(v) for v in (capture, label, family, time))
    segment = capture_segments(capture, label, family, time)
    part = np.empty(len(segment), dtype=np.int8)
    shares = np.asarray(shares, dtype=float)
    for s, e in zip(*_blocks(segment)):
        attack = label[s:e] == 1
        try:
            part[s:e] = splits.attack_rank_parts(attack, shares)
        except ValueError as err:
            raise ValueError(f"capture {capture[s]}, {err}") from err
        names = np.unique(family[s:e][attack])
        for f in names if len(names) > 1 else []:
            mine = attack & (family[s:e] == f)
            held = np.bincount(part[s:e][mine], minlength=len(shares))
            if (held < min_share * shares * mine.sum()).any():
                raise ValueError(f"capture {capture[s]} holds {held.tolist()} attacks of {f} in its parts, less than "
                                 f"{min_share} of the shares {shares.tolist()} in some part")
    return segment, part


def readme(m: dict) -> str:
    code, src, t = m["code"], m["source"], m["time_column"]
    dirty = ", which had uncommitted changes" if code["dirty"] else ""
    lines = [
        f"# LITNET-2020, processed {m['version']}",
        "",
        f"Written by `forge.data.litnet2020.prepare` on {m['created_utc']} from forge-nids commit "
        f"`{code['commit'] or 'unknown'}`{dirty}. Python {code['python']}, pandas {code['pandas']}, "
        f"pyarrow {code['pyarrow']}. {m['changes']}",
        "",
        f"`flows.parquet` holds {m['rows']} flows, {m['attacks']} of them attacks, from {len(m['captures'])} captures, "
        f"sorted by `{t}`. `manifest.json` holds the same facts in machine readable form.",
        "",
        "## Source",
        "",
        f"{src['paper']} Files from {src['repository']} at commit `{src['commit']}`. The address given in the paper, "
        "dataset.litnet.lt, redirects to that repository.",
        "",
        "| file | bytes | sha256 | used |",
        "|---|---|---|---|",
    ]
    for name, f in src["files"].items():
        used = "yes" if f["used"] else f"no, holds the same CSV as {f['same_csv_as']}, checked by size and CRC32"
        lines.append(f"| {name} | {f['bytes']} | `{f['sha256']}` | {used} |")
    lines += [
        "",
        "Every size and sha256 was checked against the downloaded file before this version was written. "
        "ALLinONE.zip, which joins the same CSV files with the ICMP and Smurf capture twice, and the "
        "`*_ATTACKERS_ONLY.zip` files, which hold the attack rows of the same captures, are not used.",
        "",
        "## Steps",
        "",
        "1. Each CSV is read without a header, with the 85 column names from the first line of `allFlows.csv` in "
        "ALLinONE.zip.",
        "2. `ts` and `te` are assembled from their six split columns and kept as written, without a time zone. The "
        "unused `tr` column shows 1970-01-01 03:00, epoch zero printed in a UTC+3 zone, so the times are most likely "
        "local Lithuanian time.",
        "3. `label` is `attack_a` and `family` maps `attack_t` one to one, see Families. Processing stops on an "
        "unknown code and on any row where the two disagree.",
        "4. `tcp_flags` is the bitmask of the six flag columns, which hold the letters UAPRSF with dots for unset "
        "bits, or `0x` and two hex digits when a bit above 0x3f is set.",
        f"5. Rows are sorted by `{t}`, then by capture, then by `row_id`, the line in the source CSV. Captures do not "
        f"overlap in `{t}`, which was checked.",
        "6. Columns are dropped as listed below. Apart from reading the ICMP and Smurf capture once, no row is "
        "filtered, deduplicated or changed.",
        "",
        "## Columns",
        "",
        "| column | type | from | role | meaning |",
        "|---|---|---|---|---|",
    ]
    for name, dtype in m["columns"].items():
        origin, role, meaning = COLUMN_NOTES[name]
        dtype = "category" if dtype.startswith("dictionary") else dtype
        lines.append(f"| `{name}` | {dtype} | {origin} | {role} | {meaning} |")
    dropped = m["dropped"]
    lines += [
        "",
        "Identifier-like features are kept so that removing them can be tested as an ablation.",
        "",
        "## Dropped columns",
        "",
        f"- {', '.join(f'`{c}`' for c in dropped['rule_columns'])}. These are the outputs of the labelling rules, "
        "`blank` on attack rows except for the rule that fired and `normal` on benign rows, so they repeat the label.",
        f"- {', '.join(f'`{c}`' for c in dropped['identifier_columns'])}. Source and destination addresses, which "
        "identify hosts and do not look anonymised.",
        "- The columns below, each with the values seen in this version.",
        "",
        "| column | values |",
        "|---|---|",
    ]
    for c, values in dropped["constant_columns"].items():
        lines.append(f"| `{c}` | {', '.join(f'`{v}`' for v in values)} |")
    lines += [
        "",
        "## Captures",
        "",
        f"In `{t}` order. The fingerprint column is the start of each capture's digest, see Fingerprint.",
        "",
        f"| capture | file | flows | attacks | prevalence | first `{t}` | last `{t}` | earliest `ts` | families | "
        "fingerprint |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in m["captures"]:
        families = ", ".join(f"{k} {v}" for k, v in c["families"].items())
        lines.append(f"| {c['capture']} | {c['file']} | {c['flows']} | {c['attacks']} | "
                     f"{c['attacks'] / c['flows']:.6f} | {c['first']} | {c['last']} | {c['ts_min']} | {families} | "
                     f"`{c['fingerprint'][:12]}` |")
    lines += [
        "",
        "## Families",
        "",
        "Attack flows per family in this version next to Table 4 of the paper. The paper counts the flows of each "
        "attack type's capture separately, while the published ICMP flood and Smurf archives hold one capture.",
        "",
        "| family | attack_t | captures | attack flows | flows of those captures | Table 4 attacks | Table 4 flows |",
        "|---|---|---|---|---|---|---|",
    ]
    for code, family in list(FAMILIES.items())[1:]:
        caps = [c for c in m["captures"] if family in c["families"]]
        paper_flows, paper_attacks = PAPER_TABLE4[family]
        lines.append(f"| {family} | `{code}` | {', '.join(c['capture'] for c in caps) or 'none'} | "
                     f"{sum(c['families'][family] for c in caps)} | {sum(c['flows'] for c in caps)} | "
                     f"{paper_attacks} | {paper_flows} |")
    lines += [
        "",
        "## Fingerprint",
        "",
        f"`{m['fingerprint']}`",
        "",
        "A sha256 over the values of every column in row order, one digest per capture, independent of how the "
        "Parquet file is encoded. `forge.data.litnet2020.fingerprint(forge.data.litnet2020.load(cfg))` recomputes it, "
        "so a table built again from the same files can be checked against this one.",
        "",
    ]
    return "\n".join(lines)
