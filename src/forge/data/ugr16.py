"""UGR'16, the netflow records of a Spanish ISP as published by its authors, the six test weeks of July and August
2016 with every k-th flow kept in file order and written as one time ordered table.

the authors collected netflow v9 records at the border of a Spanish ISP from March to August 2016 and published them
in weekly archives. the calibration set, March to June, holds real background traffic with the flows of blacklisted
addresses and the spam they found marked. the test set, the end of July to the end of August, holds the same
background with attacks the authors launched from machines inside the network, a DoS, the port scans scan11 and
scan44, named after their one attacker against one victim and four against four, and the Neris botnet, beside the
real anomalies they found, spam, SSH scans and UDP scans, and the blacklisted addresses again. every flow is one line
of 13 comma separated fields without a header, end time, duration, source and destination address, source and
destination port, protocol, TCP flags, forwarding status, type of service, packets, bytes and label, and each week is
one tar.gz holding one such file with the unique flows and the blacklist flows marked.

the official site has returned 403 since the middle of 2026, so the archives come from the Hugging Face dataset
mehedi107/ugr-16, which holds the official weekly archives unchanged with a manifest of the official address, the
size and the sha256 of each. no checksum was ever published on the official site, so the sha256 pinned below is the
mirror's, and the archive of the last test week was checked on 2026-10-17 against the official file name, its owner
and its layout. the test weeks hold about 3.7 billion flows, fifty times the sessions of AnoShift, so prepare keeps
every k-th flow of a week in file order, the systematic sample that keeps the order, the bursts and the label shares at
a coarser grain, and records the full count of every label beside the sampled ones. a few lines of the archives were
cut and run into the line after them, so every field is read as text and the kept rows are cast afterwards, a row
holding a value that cannot be read dropped and counted. the archives are fetched one at a time into scratch space and
deleted once their week is written, since 57 GB would not fit on Drive, and a week already written is kept across
interrupted runs. download fetches nothing in that mode and the raw archives otherwise.
"""

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from forge import paths
from forge.data import digest
from forge.data.litnet2020 import flag_value
from forge.runs import git_state

PAPER = (
    "Maciá-Fernández G., Camacho J., Magán-Carrión R., García-Teodoro P., Therón R. UGR'16: A new dataset for "
    "the evaluation of cyclostationarity-based network IDSs. Computers & Security 73:411-424, 2018. "
    "doi:10.1016/j.cose.2017.11.004."
)
SITE = "https://nesg.ugr.es/nesg-ugr16/"
MIRROR = "mehedi107/ugr-16"
URL = f"https://huggingface.co/datasets/{MIRROR}/resolve/main/{{path}}"
# the test weeks in time order, each with its path in the mirror, which repeats the official path under
# download/attack, and its size and sha256 as the manifest of the mirror recorded them, read on 2026-10-17
FILES = {
    "july_week5": ("test/july/week5/july_week5_csv.tar.gz", 7709006531,
                   "69ad4f8d56d6602cf5f7daef77fbc4d851021f96a9097865d34affa2abcc23db"),
    "august_week1": ("test/august/week1/august_week1_csv.tar.gz", 12079340169,
                     "1650d31d6ee5d07df6ca64467092f6929134681e420ab174f3dad84b91ee5b5e"),
    "august_week2": ("test/august/week2/august_week2_csv.tar.gz", 12060990744,
                     "2e92eba04e7355afa02d711bdb2eb1b06ea8e3463c14c884700331aa93a0b34b"),
    "august_week3": ("test/august/week3/august_week3_csv.tar.gz", 11856637347,
                     "defc3162ced2010679229640826398fbfd8c09f39e024339dc8afc3eb4e5ffbf"),
    "august_week4": ("test/august/week4/august_week4_csv.tar.gz", 12505318573,
                     "beec73f74cb98ef871b175288131b2f3f361480e7f0c741dae424e60c03a2ae4"),
    "august_week5": ("test/august/week5/august_week5_csv.tar.gz", 571031116,
                     "6f71042f8259a5768e26fd7d0c545c5916ae079dcec72cbf0e9fb570dedb63e7"),
}
WEEK_ORDER = list(FILES)

# what each processed version changed, written into its readme
VERSIONS = {"v1": "First version."}

# the fields of a line, in order, and the ones kept. addresses identify hosts and the attacks and the blacklist are
# labelled by address, so they stay in the archives
RAW_COLUMNS = ["te", "td", "sa", "da", "sp", "dp", "pr", "flg", "fwd", "stos", "ipkt", "ibyt", "label"]
IDENTIFIER_COLUMNS = ("sa", "da")
READ_COLUMNS = [c for c in RAW_COLUMNS if c not in IDENTIFIER_COLUMNS]
# the end time arrives as 2016-08-29 00:12:21 and the ports as whole numbers for every protocol, ICMP included, which
# the last test week showed on every row. every field is read as text and the kept rows are cast to these types in
# _cast, since the archives hold a few cut lines that run into the line after them and the csv reader would stop the
# week on the first value it cannot convert
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
RAW_TYPES = {
    "te": pa.timestamp("s"),
    "td": pa.float64(),
    "sp": pa.int64(),
    "dp": pa.int64(),
    "pr": pa.string(),
    "flg": pa.string(),
    "fwd": pa.int64(),
    "stos": pa.int64(),
    "ipkt": pa.int64(),
    "ibyt": pa.int64(),
    "label": pa.string(),
}
# the range of each integer field, a port, a byte or a count, so a cut line that pushed a port into the millions is
# dropped rather than narrowed, and the digits an integer may have, so no cast can overflow int64
RANGES = {"sp": (0, 65535), "dp": (0, 65535), "fwd": (0, 255), "stos": (0, 255), "ipkt": (0, None), "ibyt": (0, None)}
MAX_DIGITS = 18
# a duration as nfdump writes it, seconds with three decimals
DURATION = r"^[0-9]{1,12}(\.[0-9]{1,6})?$"
# the share of the sampled rows of a week that may be dropped for a value that cannot be read. the cut lines are a few
# in a hundred million, so more than one in a hundred means the fields are not what the columns say
MAX_DROPPED = 0.01
# dropped rows recorded in the summary of a week with their line, the field that failed and its text
EXAMPLES = 5

# the labels the authors give and the family name used here. a label outside this list is kept under its own name,
# lower case with underscores, and reported, since the weeks are read once and a stop would cost the week
LABELS = {
    "background": "benign",
    "dos": "dos",
    "scan11": "scan11",
    "scan44": "scan44",
    "nerisbotnet": "nerisbotnet",
    "anomaly-spam": "anomaly_spam",
    "anomaly-sshscan": "anomaly_sshscan",
    "anomaly-udpscan": "anomaly_udpscan",
    "blacklist": "blacklist",
}
FAMILY_ORDER = list(LABELS.values())
ATTACK_FAMILIES = FAMILY_ORDER[1:]
# the attacks the authors launched, against the anomalies they found in the real traffic
INJECTED_FAMILIES = ["dos", "scan11", "scan44", "nerisbotnet"]
# family names as figures show them
FAMILY_LABELS = {
    "benign": "benign",
    "dos": "DoS",
    "scan11": "port scan, one to one",
    "scan44": "port scan, four to four",
    "nerisbotnet": "Neris botnet",
    "anomaly_spam": "spam",
    "anomaly_sshscan": "SSH scan",
    "anomaly_udpscan": "UDP scan",
    "blacklist": "blacklisted address",
}

# Parquet has no second unit, so times are stored in ms and keep their whole second values
SCHEMA = pa.schema([
    ("te", pa.timestamp("ms")),
    ("week", pa.dictionary(pa.int8(), pa.string())),
    ("row_id", pa.int32()),
    ("label", pa.int8()),
    ("family", pa.dictionary(pa.int8(), pa.string())),
    ("td", pa.float64()),
    ("proto", pa.dictionary(pa.int16(), pa.string())),
    ("tcp_flags", pa.uint8()),
    ("fwd", pa.uint8()),
    ("stos", pa.uint8()),
    ("ipkt", pa.int64()),
    ("ibyt", pa.int64()),
    ("sp", pa.int32()),
    ("dp", pa.int32()),
])
# source and meaning of each column of the processed table, written into the readme
COLUMN_NOTES = {
    "te": ("te", "time", "flow end, as written, 1 s resolution"),
    "week": ("archive name", "provenance", "test week the flow comes from"),
    "row_id": ("line", "provenance", "position of the flow among the parsed lines of its week, counted from 1"),
    "label": ("label", "label", "1 attack, anomaly or blacklisted address, 0 background"),
    "family": ("label", "label", "attack family, benign for background flows"),
    "td": ("td", "feature", "duration in seconds"),
    "proto": ("pr", "feature", "protocol name as written by nfdump, or its number when it has no name"),
    "tcp_flags": ("flg", "feature", "TCP flag bitmask, CWR 0x80 down to FIN 0x01"),
    "fwd": ("fwd", "feature", "forwarding status"),
    "stos": ("stos", "feature", "source type of service byte"),
    "ipkt": ("ipkt", "feature", "packets"),
    "ibyt": ("ibyt", "feature", "bytes"),
    "sp": ("sp", "identifier-like feature", "source port"),
    "dp": ("dp", "feature", "destination port, ICMP type and code for ICMP"),
}

# bytes of csv text handed to the parser at a time, and bytes read from the network and written to disk at a time
CSV_BLOCK = 64 << 20
BLOCK = 1 << 20
ROW_GROUP_ROWS = 1_000_000
SAMPLE_META = "sample.json"


def raw_dir(cfg: dict) -> Path:
    return paths.data_dir("raw") / cfg["dataset"]["raw_dir"]


def processed_dir(cfg: dict) -> Path:
    d = cfg["dataset"]
    return paths.data_dir("processed") / d["name"] / d["processed_version"]


def scratch_dir(cfg: dict) -> Path:
    # local disk, where an archive of up to 15 GB lives only while its week is read
    d = cfg["dataset"]
    if not d.get("raw_scratch", False):
        return raw_dir(cfg)
    return Path(tempfile.gettempdir()) / paths.PROJECT_NAME / "raw" / d["raw_dir"]


def official_path(path: str) -> str:
    # the mirror keeps the official folder layout under test, where the site had download/attack
    return "download/attack/" + path.removeprefix("test/")


def weeks_of(cfg: dict, files: dict) -> list[str]:
    weeks = list(cfg["dataset"].get("weeks", files))
    unknown = [w for w in weeks if w not in files]
    if unknown:
        raise ValueError(f"unknown weeks {', '.join(unknown)}, the archives are {', '.join(files)}")
    return sorted(weeks, key=list(files).index)


def sample_every(cfg: dict) -> int:
    every = int(cfg["dataset"].get("sample_every", 1))
    if every < 1:
        raise ValueError(f"sample_every is {every}, expected 1 or more")
    return every


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(BLOCK):
            h.update(block)
    return h.hexdigest()


def fetch(path: Path, url: str, size: int, sha256: str, log=print) -> Path:
    # an archive gets its final name only after its size and sha256 match, so a name in the folder means a checked copy
    if path.exists() and path.stat().st_size == size:
        if _sha256(path) == sha256:
            log(f"found {path.name}, {size} bytes, sha256 matches")
            return path
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    log(f"fetching {url}, {size} bytes")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as f:
        while block := response.read(BLOCK):
            f.write(block)
            h.update(block)
    if part.stat().st_size != size or h.hexdigest() != sha256:
        part.unlink()
        raise ValueError(f"{path.name} arrived with a size or sha256 other than the mirror's and was removed")
    part.replace(path)
    log(f"fetched {path.name}, {size} bytes, sha256 matches")
    return path


def download(cfg: dict, log=print, files: dict | None = None, url: str = URL) -> Path:
    # with raw_scratch the archives are fetched one at a time by prepare and deleted, so there is nothing to keep
    files = FILES if files is None else files
    folder = scratch_dir(cfg)
    if cfg["dataset"].get("raw_scratch", False):
        log(f"raw_scratch is set, prepare fetches each archive into {folder} and deletes it after its week")
        return folder
    for week in weeks_of(cfg, files):
        path, size, sha256 = files[week]
        fetch(folder / Path(path).name, url.format(path=path), size, sha256, log)
    return folder


def family_name(label: str) -> str:
    return LABELS.get(label, label.strip().lower().replace("-", "_").replace(" ", "_") or "unlabelled")


def _fit(values, dtype, name: str) -> np.ndarray:
    # narrows an integer column only when every value fits, so nothing wraps around silently
    values = np.asarray(values)
    info = np.iinfo(dtype)
    if values.size and (values.min() < info.min or values.max() > info.max):
        raise ValueError(f"{name} holds values outside {info.min} to {info.max}")
    return values.astype(dtype)


def _strings(values: pa.Array) -> list[str]:
    # the lines are taken as they are, so a byte outside utf-8 is replaced here rather than stopping the week
    return [b.decode("utf-8", "replace") for b in values.cast(pa.binary()).to_pylist()]


def _parsed_dictionary(column: pa.ChunkedArray, parse, dtype) -> np.ndarray:
    # a few hundred distinct strings at most, so each is parsed once and spread over its rows
    encoded = column.combine_chunks().dictionary_encode()
    values = np.array([parse(t) for t in _strings(encoded.dictionary)], dtype=dtype)
    return values[encoded.indices.to_numpy(zero_copy_only=False)]


def _categorical(column: pa.ChunkedArray, rename=lambda name: name, first: list[str] = ()) -> pd.Categorical:
    # the distinct strings of the column as categories, the ones listed in first in that order and the rest sorted
    encoded = column.combine_chunks().dictionary_encode()
    names = [rename(t) for t in _strings(encoded.dictionary)]
    order = [n for n in first if n in names] + sorted(set(names) - set(first))
    remap = np.array([order.index(n) for n in names], dtype=np.int16)
    codes = remap[encoded.indices.to_numpy(zero_copy_only=False)] if len(names) else np.array([], dtype=np.int16)
    return pd.Categorical.from_codes(codes, categories=order)


def clean(raw: pa.Table, week: str, weeks: list[str], row_id: np.ndarray) -> pd.DataFrame:
    family = _categorical(raw.column("label"), family_name, FAMILY_ORDER)
    te = raw.column("te").combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[s]")
    return pd.DataFrame({
        "te": te,
        "week": pd.Categorical.from_codes(np.full(raw.num_rows, weeks.index(week)), categories=weeks),
        "row_id": _fit(row_id, np.int32, "row_id"),
        "label": (family.codes != list(family.categories).index("benign")).astype(np.int8)
        if "benign" in family.categories else np.ones(raw.num_rows, dtype=np.int8),
        "family": family,
        "td": raw.column("td").to_numpy().astype(np.float64),
        "proto": _categorical(raw.column("pr")),
        "tcp_flags": _parsed_dictionary(raw.column("flg"), flag_value, np.uint8),
        "fwd": _fit(raw.column("fwd").to_numpy(), np.uint8, "fwd"),
        "stos": _fit(raw.column("stos").to_numpy(), np.uint8, "stos"),
        "ipkt": raw.column("ipkt").to_numpy().astype(np.int64),
        "ibyt": raw.column("ibyt").to_numpy().astype(np.int64),
        "sp": _fit(raw.column("sp").to_numpy(), np.int32, "sp"),
        "dp": _fit(raw.column("dp").to_numpy(), np.int32, "dp"),
    })


def _cast(raw: pa.RecordBatch, row_id: np.ndarray, limit: int) -> tuple[pa.Table, np.ndarray, int, list[dict]]:
    """the kept rows cast from text to their raw types, without the rows holding a value that cannot be read.

    a cut line of the archives runs into the line after it, so a port can read as a port followed by a timestamp.
    every field arrives as text and is checked before its cast, a time by parsing it, a count by its digits, its
    length and its range, a duration by its shape, and a row that fails any check is dropped. the dropped rows come
    back as a count and, up to the limit and in line order, as their line, the first field that failed and its text."""
    columns, ok, found = {}, np.ones(raw.num_rows, dtype=bool), []
    for name in READ_COLUMNS:
        text, dtype = raw.column(name), RAW_TYPES[name]
        if dtype == pa.string():
            columns[name] = text
            continue
        text = pc.ascii_trim_whitespace(text)
        if name == "te":
            values = pc.strptime(text, format=TIME_FORMAT, unit="s", error_is_null=True)
            good = values.is_valid()
        elif name == "td":
            good = pc.match_substring_regex(text, DURATION)
            values = pc.cast(pc.if_else(good, text, pa.scalar(None, pa.string())), dtype)
        else:
            low, high = RANGES[name]
            good = pc.and_(pc.ascii_is_decimal(text), pc.less_equal(pc.binary_length(text), MAX_DIGITS))
            values = pc.cast(pc.if_else(good, text, pa.scalar(None, pa.string())), dtype)
            good = pc.greater_equal(values, low) if high is None else pc.and_(pc.greater_equal(values, low),
                                                                              pc.less_equal(values, high))
        good = pc.fill_null(good, False).to_numpy(zero_copy_only=False)
        # the earliest failures of each field, among the rows no earlier field failed, hold the earliest of the row
        failed = np.flatnonzero(ok & ~good)[:max(limit, 0)]
        found += [(int(i), name, v) for i, v in zip(failed, _strings(text.take(pa.array(failed))))]
        ok &= good
        columns[name] = values
    examples = [{"row_id": int(row_id[i]), "column": name, "value": v} for i, name, v in sorted(found)[:limit]]
    return pa.table(columns).filter(pa.array(ok)), row_id[ok], int((~ok).sum()), examples


def _member(tar: tarfile.TarFile) -> tarfile.TarInfo:
    # the archive is read as a stream, so the first regular file is taken and must be the only one
    member = next((m for m in tar if m.isfile()), None)
    if member is None:
        raise ValueError("the archive holds no file")
    return member


def read_week(path: Path, week: str, weeks: list[str], every: int, log=print) -> tuple[pd.DataFrame, dict]:
    """every k-th flow of one archive in file order with the counts of all its lines.

    the csv is streamed through the arrow reader in blocks with every field as text, and only the kept rows of each
    block are checked and cast, so the cost of a week is its decompression and the label counts. lines with another
    number of fields are skipped and counted, empty lines are ignored, a kept row holding a value that cannot be read
    is dropped and counted, and the flows are sorted by their end time with ties in file order, which the archives
    already satisfy."""
    skipped = {"n": 0}

    def skip(row):
        skipped["n"] += 1
        return "skip"

    read = pcsv.ReadOptions(column_names=RAW_COLUMNS, block_size=CSV_BLOCK)
    parse = pcsv.ParseOptions(invalid_row_handler=skip)
    # a byte outside utf-8 is left in place, since a check would stop the week and the text is cast or decoded later
    convert = pcsv.ConvertOptions(column_types={c: pa.string() for c in RAW_COLUMNS}, include_columns=READ_COLUMNS,
                                  check_utf8=False)
    kept, row_ids, counts, rows, dropped, examples = [], [], {}, 0, 0, []
    with tarfile.open(path, "r|gz") as tar:
        member = _member(tar)
        with tar.extractfile(member) as f, pcsv.open_csv(f, read_options=read, parse_options=parse,
                                                          convert_options=convert) as reader:
            for batch in reader:
                n = batch.num_rows
                seen = pc.value_counts(batch.column("label"))
                for value, count in zip(_strings(seen.field(0)), seen.field(1).to_pylist()):
                    counts[value] = counts.get(value, 0) + count
                # rows are numbered from 1 over the whole week, and row 1, 1 + k, 1 + 2k and on are kept
                first = (-rows) % every
                indices = np.arange(first, n, every)
                if len(indices):
                    table, ids, bad, found = _cast(batch.take(pa.array(indices)), rows + indices + 1,
                                                   EXAMPLES - len(examples))
                    kept.append(table)
                    row_ids.append(ids)
                    dropped += bad
                    examples += found
                rows += n
        # the stream stands after the csv, and next reads the headers that follow, since iterating the archive again
        # would start from the members already seen
        extra = []
        while (m := tar.next()) is not None:
            if m.isfile():
                extra.append(m.name)
    if extra:
        raise ValueError(f"{path.name} holds {', '.join(extra)} beside {member.name}, one csv was expected")
    if rows == 0:
        raise ValueError(f"{path.name} holds no flow")
    sampled = sum(t.num_rows for t in kept) + dropped
    if dropped > MAX_DROPPED * sampled:
        e = examples[0]
        raise ValueError(f"{path.name} holds {dropped} sampled rows of {sampled} with a value that cannot be read, "
                         f"far more than its few cut lines, e.g. line {e['row_id']} {e['column']} {e['value']!r}")
    in_week = {}
    for value, count in counts.items():
        in_week[family_name(value)] = in_week.get(family_name(value), 0) + int(count)
    table = pa.concat_tables(kept) if kept else pa.table({c: pa.array([], RAW_TYPES[c]) for c in READ_COLUMNS})
    flows = clean(table, week, weeks, np.concatenate(row_ids) if row_ids else np.array([], dtype=np.int64))
    in_order = bool(flows["te"].is_monotonic_increasing)
    # ties in time keep the order of the source file, which row_id records
    flows = flows.sort_values(["te", "row_id"], kind="stable", ignore_index=True)
    attacks = flows["label"].to_numpy() == 1
    families = flows.loc[attacks, "family"].value_counts()
    summary = {
        "week": week,
        "file": path.name,
        "member": member.name,
        "member_bytes": int(member.size),
        "lines": int(rows + skipped["n"]),
        "skipped_lines": int(skipped["n"]),
        "flows_in_week": int(rows),
        "labels_in_week": dict(sorted(in_week.items())),
        "every": int(every),
        "flows": int(len(flows)),
        "dropped": int(dropped),
        "dropped_examples": examples,
        "attacks": int(attacks.sum()),
        "in_order": in_order,
        "first": str(flows["te"].iloc[0]),
        "last": str(flows["te"].iloc[-1]),
        "families": {str(k): int(v) for k, v in families.items() if v > 0},
        "unknown_labels": sorted(set(counts) - set(LABELS)),
        "fingerprint": digest.table_digest(flows, SCHEMA),
    }
    log(f"read {path.name}, {rows} flows in the week, {len(flows)} kept, {summary['attacks']} attacks, te from "
        f"{summary['first']} to {summary['last']}" + (f", {skipped['n']} lines skipped" if skipped["n"] else "")
        + (f", {dropped} kept rows dropped for a value that cannot be read, e.g. line {examples[0]['row_id']} "
           f"{examples[0]['column']} {examples[0]['value']!r}" if dropped else "")
        + (f", unknown labels {', '.join(summary['unknown_labels'])}" if summary["unknown_labels"] else ""))
    return flows, summary


def _table(flows: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(flows, schema=SCHEMA, preserve_index=False)


def _in_order(table: pa.Table, weeks: list[str]) -> pa.Table:
    # the rows of one boundary, a few thousand, sorted by te, then by week in time order, then by row_id
    frame = table.unify_dictionaries().to_pandas()
    rank = frame["week"].astype(str).map(weeks.index).to_numpy()
    order = np.lexsort((frame["row_id"].to_numpy(), rank, frame["te"].to_numpy()))
    return _table(frame.iloc[order].reset_index(drop=True))


def _write_in_order(writer: pq.ParquetWriter, parts: Path, summaries: list[dict], weeks: list[str]) -> list[dict]:
    """writes the weeks as one table in te order, the rows of each boundary merged, and describes the boundaries.

    an archive holds the flows exported in its week and the export trails the end of a flow by seconds, so the first
    flows of a week end before the last flows of the week before it, by a minute at most. the rows of a boundary, the
    tail of the earlier week from the first te of the later one and the head of the later week up to the last te of
    the earlier one, are sorted together by te, week and row_id, and the rest of each week is written as it is, which
    keeps every week in its own order and costs one week of memory at a time."""
    boundaries, tail, before = [], None, None
    for s, after in zip(summaries, summaries[1:] + [None]):
        table = pq.read_table(parts / f"{s['week']}.parquet")
        if before is not None:
            last = pa.scalar(datetime.fromisoformat(before["last"]), pa.timestamp("ms"))
            head = table.filter(pc.less_equal(table.column("te"), last))
            table = table.filter(pc.greater(table.column("te"), last))
            merged = pa.concat_tables([tail, head])
            if merged.num_rows:
                writer.write_table(_in_order(merged, weeks), row_group_size=ROW_GROUP_ROWS)
            overlap = datetime.fromisoformat(before["last"]) - datetime.fromisoformat(s["first"])
            boundaries.append({"weeks": [before["week"], s["week"]], "overlap_seconds": overlap.total_seconds(),
                               "rows": [tail.num_rows, head.num_rows]})
        if after is not None:
            first = pa.scalar(datetime.fromisoformat(after["first"]), pa.timestamp("ms"))
            tail = table.filter(pc.greater_equal(table.column("te"), first))
            table = table.filter(pc.less(table.column("te"), first))
        writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
        before = s
    return boundaries


def prepare(cfg: dict, log=print, files: dict | None = None, url: str = URL, scratch: Path | None = None) -> Path:
    files = FILES if files is None else files
    d = cfg["dataset"]
    version, t = d["processed_version"], d.get("time_column", "te")
    if version not in VERSIONS:
        raise ValueError(f"VERSIONS has no description of {version}, the readme of a version says what it changed")
    if t != "te":
        raise ValueError(f"time_column is te, the only time the archives hold, not {t}")
    weeks, every = weeks_of(cfg, files), sample_every(cfg)
    delete = bool(d.get("raw_scratch", False))
    scratch = scratch_dir(cfg) if scratch is None else scratch
    out = processed_dir(cfg)
    if out.exists():
        raise FileExistsError(f"{out} exists and a processed version is never overwritten, name a new version")

    # a version folder appears under its final name only once complete. the parts of the weeks already written stay
    # in the partial folder across interrupted runs, since a week costs a download of 12 GB and its reading
    partial = out.with_name(f"{out.name}.partial")
    parts = partial / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    summaries = []
    for week in weeks:
        path, size, sha256 = files[week]
        part, meta = parts / f"{week}.parquet", parts / f"{week}.json"
        if part.exists() and meta.exists():
            s = json.loads(meta.read_text(encoding="utf-8"))
            if s.get("sha256") == sha256 and s.get("every") == every:
                # a part written before the fields were read as text has no dropped count, and that reader stopped
                # on any value it could not cast, so its count is 0
                s.setdefault("dropped", 0)
                s.setdefault("dropped_examples", [])
                log(f"found {week} in {parts}, {s['flows']} flows kept of {s['flows_in_week']}")
                summaries.append(s)
                continue
            part.unlink()
            meta.unlink()
        archive = fetch(scratch / Path(path).name, url.format(path=path), size, sha256, log)
        flows, s = read_week(archive, week, weeks, every, log)
        s["sha256"] = sha256
        tmp = part.with_name(f"{part.name}.part")
        pq.write_table(_table(flows), tmp, compression="zstd", row_group_size=ROW_GROUP_ROWS)
        tmp.replace(part)
        meta.write_text(json.dumps(s, indent=2), encoding="utf-8")
        summaries.append(s)
        del flows
        if delete:
            archive.unlink()
            log(f"deleted {archive.name}")
    # the weeks overlap in te at their boundaries by a minute at most, see _write_in_order, which needs each week to
    # start and end after the one before it and to stay clear of the week after next
    for a, b in zip(summaries, summaries[1:]):
        if not (a["first"] < b["first"] and a["last"] < b["last"]):
            raise ValueError(f"weeks {a['week']} and {b['week']} are not in te order, {a['first']} to {a['last']} "
                             f"against {b['first']} to {b['last']}")
    for a, c in zip(summaries, summaries[2:]):
        if not a["last"] < c["first"]:
            raise ValueError(f"weeks {a['week']} and {c['week']} overlap in te although a week lies between them, so "
                             "the boundaries cannot be merged one at a time")
    with pq.ParquetWriter(partial / "flows.parquet", SCHEMA, compression="zstd") as writer:
        boundaries = _write_in_order(writer, parts, summaries, weeks)

    manifest = {
        "dataset": d["name"],
        "version": version,
        "changes": VERSIONS[version],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {**git_state(), "python": sys.version.split()[0], "pandas": pd.__version__, "pyarrow": pa.__version__},
        "source": {
            "paper": PAPER,
            "site": SITE,
            "mirror": f"https://huggingface.co/datasets/{MIRROR}",
            "files": {
                w: {"path": files[w][0], "official_path": official_path(files[w][0]), "bytes": files[w][1],
                    "sha256": files[w][2]}
                for w in weeks
            },
        },
        "time_column": t,
        "sample_every": every,
        "rows": sum(s["flows"] for s in summaries),
        "attacks": sum(s["attacks"] for s in summaries),
        "rows_in_weeks": sum(s["flows_in_week"] for s in summaries),
        "columns": {f.name: str(f.type) for f in SCHEMA},
        "dropped": {"identifier_columns": list(IDENTIFIER_COLUMNS), "rows": sum(s["dropped"] for s in summaries)},
        "weeks": summaries,
        "boundaries": boundaries,
        "fingerprint": digest.combine((s["week"], s["fingerprint"]) for s in summaries),
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (partial / "readme.md").write_text(readme(manifest), encoding="utf-8")
    shutil.rmtree(parts)
    partial.rename(out)
    log(f"wrote {manifest['rows']} flows to {out}, every {every}th of {manifest['rows_in_weeks']}, fingerprint "
        f"{manifest['fingerprint']}")
    return out


def read_manifest(cfg: dict) -> dict:
    return json.loads((processed_dir(cfg) / "manifest.json").read_text(encoding="utf-8"))


def load(cfg: dict, columns: list[str] | None = None, weeks: list[str] | None = None) -> pd.DataFrame:
    # the processed table in the order it was written, by flow end with ties by week and then by row_id, only the
    # given weeks when weeks is set. the known families come first in the category order, any other label after them
    filters = None if weeks is None else [("week", "in", list(weeks))]
    flows = pq.read_table(processed_dir(cfg) / "flows.parquet", columns=columns, filters=filters).to_pandas()
    if "family" in flows.columns:
        seen = list(flows["family"].cat.categories)
        flows["family"] = flows["family"].cat.set_categories(FAMILY_ORDER + sorted(set(seen) - set(FAMILY_ORDER)))
    if "week" in flows.columns:
        present = list(flows["week"].cat.categories)
        flows["week"] = flows["week"].cat.set_categories([w for w in WEEK_ORDER if w in present]
                                                         + [w for w in present if w not in WEEK_ORDER])
    return flows


def fingerprint(flows: pd.DataFrame) -> str:
    # recomputes the fingerprint of a full table read back with load, one digest per week over its rows in table order,
    # since the weeks interleave at their boundaries and form no blocks
    weeks = [w for w in flows["week"].cat.categories if (flows["week"] == w).any()]
    return digest.combine((w, digest.table_digest(flows[flows["week"] == w], SCHEMA)) for w in weeks)


def readme(m: dict) -> str:
    code, src = m["code"], m["source"]
    dirty = ", which had uncommitted changes" if code["dirty"] else ""
    k = m["sample_every"]
    lines = [
        f"# UGR'16, processed {m['version']}",
        "",
        f"Written by `forge.data.ugr16.prepare` on {m['created_utc']} from forge-nids commit "
        f"`{code['commit'] or 'unknown'}`{dirty}. Python {code['python']}, pandas {code['pandas']}, "
        f"pyarrow {code['pyarrow']}. {m['changes']}",
        "",
        f"`flows.parquet` holds {m['rows']} flows, {m['attacks']} of them attacks, anomalies or blacklisted "
        f"addresses, every {k}th flow in file order of the {m['rows_in_weeks']} flows of {len(m['weeks'])} test "
        f"{'week' if len(m['weeks']) == 1 else 'weeks'}, sorted by `te` with ties by `week` and `row_id`. "
        "`manifest.json` holds the same facts in machine readable form.",
        "",
        "## Source",
        "",
        f"{src['paper']} The official site is {src['site']}, which has returned 403 since the middle of 2026. The "
        f"archives come from {src['mirror']}, which holds the official weekly archives unchanged with a manifest of "
        "the official address, the size and the sha256 of each. The official site never published a checksum, so the "
        "sha256 below is the mirror's, and the last test week was checked against the official file name, its owner "
        "and its layout on 2026-10-17.",
        "",
        "| week | official path | bytes | sha256 |",
        "|---|---|---|---|",
    ]
    for week, f in src["files"].items():
        lines.append(f"| {week} | {f['official_path']} | {f['bytes']} | `{f['sha256']}` |")
    lines += [
        "",
        "Every size and sha256 was checked against the fetched archive before its week was read. The calibration "
        "weeks of March to June, the per attack subsets, the nfcapd files and the attack timestamp files are not used.",
        "",
        "## Steps",
        "",
        "1. Each archive is streamed once and its one csv read without a header, with the 13 field names of the "
        "authors' description. The addresses are not read.",
        f"2. The flows are numbered from 1 in file order and every {k}th one is kept, counting from the first, so "
        "the order, the bursts and the label shares survive at a coarser grain. Lines with another number of fields "
        "are skipped and counted, empty lines ignored. A kept flow holding a value that cannot be read, which the "
        "archives have where a line was cut and runs into the next, is dropped and counted with its line, field and "
        "text. The label counts of every line of the week are recorded beside the counts of the kept flows.",
        "3. `label` is 1 for every flow not labelled background and `family` maps the authors' label, see Families. A "
        "label outside the known list is kept under its own name and reported.",
        "4. `tcp_flags` is the bitmask of the six flag letters UAPRSF with dots for unset bits, or `0x` and two hex "
        "digits when a bit above 0x3f is set.",
        "5. Rows are sorted by `te`, then by `week`, then by `row_id`, the position in the week, which the archives "
        "already satisfy within a week. An archive holds the flows exported in its week and the export trails the "
        "end of a flow by seconds, so the first flows of a week end before the last flows of the week before it, by "
        "a minute at most, and the rows of each boundary are merged in `te` order, see Boundaries.",
        "6. Apart from the sampling, no row is filtered, deduplicated or changed.",
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
    lines += [
        "",
        f"Dropped: {', '.join(f'`{c}`' for c in m['dropped']['identifier_columns'])}, the source and destination "
        "addresses, which identify hosts and carry the labels of the attacks and of the blacklist"
        + (f", and {m['dropped']['rows']} kept flows with a value that cannot be read, listed under Weeks."
           if m["dropped"].get("rows") else "."),
        "",
        "## Weeks",
        "",
        "In `te` order. Flows and labels of the week count every line of the archive, the kept columns the sample "
        "without the dropped flows, which held a value that could not be read. The fingerprint column is the start "
        "of each week's digest, see Fingerprint.",
        "",
        "| week | file | flows of the week | kept | attacks kept | prevalence | first `te` | last `te` | in file "
        "order | skipped lines | dropped | families kept | fingerprint |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for w in m["weeks"]:
        families = ", ".join(f"{k} {v}" for k, v in w["families"].items()) or "none"
        lines.append(f"| {w['week']} | {w['file']} | {w['flows_in_week']} | {w['flows']} | {w['attacks']} | "
                     f"{w['attacks'] / max(w['flows'], 1):.6f} | {w['first']} | {w['last']} | "
                     f"{'yes' if w['in_order'] else 'no'} | {w['skipped_lines']} | {w.get('dropped', 0)} | "
                     f"{families} | `{w['fingerprint'][:12]}` |")
    examples = [(w["week"], e) for w in m["weeks"] for e in w.get("dropped_examples", [])]
    if examples:
        lines += ["", f"Dropped flows, the first {EXAMPLES} of each week by line, field and text:", ""]
        lines += [f"- {week}, line {e['row_id']}, {e['column']} `{e['value']}`" for week, e in examples]
    if m.get("boundaries"):
        lines += [
            "",
            "## Boundaries",
            "",
            "The overlap in `te` between each week and the one before it, in seconds, with the rows merged at the "
            "boundary, those of the earlier week from the first `te` of the later one and those of the later week up "
            "to the last `te` of the earlier one. A negative overlap is a gap.",
            "",
            "| earlier week | later week | overlap in seconds | rows of the earlier week | rows of the later week |",
            "|---|---|---|---|---|",
        ]
        for b in m["boundaries"]:
            lines.append(f"| {b['weeks'][0]} | {b['weeks'][1]} | {b['overlap_seconds']:.0f} | {b['rows'][0]} | "
                         f"{b['rows'][1]} |")
    lines += [
        "",
        "## Families",
        "",
        "Flows per label over every line of each week, before the sampling, with the family name used here. The "
        "attacks dos, scan11, scan44 and nerisbotnet were launched by the authors, the anomalies were found by them "
        "in the real traffic, and blacklist marks flows of addresses on public blacklists.",
        "",
        "| label | family | " + " | ".join(w["week"] for w in m["weeks"]) + " |",
        "|---|---|" + "---|" * len(m["weeks"]),
    ]
    labels = {}
    for w in m["weeks"]:
        for family in w["labels_in_week"]:
            labels.setdefault(family, None)
    # the authors' spelling of every label, the known ones from the mapping and the others as the weeks reported them
    names = {v: k for k, v in LABELS.items()}
    names.update({family_name(label): label for w in m["weeks"] for label in w["unknown_labels"]})
    for family in [f for f in FAMILY_ORDER if f in labels] + sorted(set(labels) - set(FAMILY_ORDER)):
        counts = " | ".join(str(w["labels_in_week"].get(family, 0)) for w in m["weeks"])
        lines.append(f"| {names.get(family, family)} | {family} | {counts} |")
    lines += [
        "",
        "## Fingerprint",
        "",
        f"`{m['fingerprint']}`",
        "",
        "A sha256 over the values of every column in row order, one digest per week, independent of how the Parquet "
        "file is encoded. `forge.data.ugr16.fingerprint(forge.data.ugr16.load(cfg))` recomputes it, so a table built "
        "again from the same archives can be checked against this one.",
        "",
    ]
    return "\n".join(lines)
