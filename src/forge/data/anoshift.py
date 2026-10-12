"""AnoShift, the Kyoto 2006+ honeypot sessions of 2006 to 2015 as preprocessed by its authors, the ten yearly test
files written as one table in year and original row order.

Kyoto 2006+ holds session records from the honeypots of Kyoto University, labelled 1 normal, -1 known attack and -2
unknown attack by its authors. AnoShift publishes them as one tar on Google storage, with a train and a test parquet
file for each year from 2006, which starts in November, to 2015. the test file of a year holds a random 10% of each
label class of that year and the train file the rest. both keep Kyoto fields 0 to 13, with the session duration and
the two byte counts in log bins, and the label in column 18, but no time stamp, address or port. the row index of a
year runs from 0 through its train and test file in the order of the daily source files, so sorting the test file by
it gives a 10% sample of every class in the original order of the year, without clock time.

download streams the tar once, hashes all of it and keeps only the ten test files, each hashed as well. no hash is
published with the tar, so the one of the first download, on 2026-10-12, is pinned below and a later download of a
changed tar stops. prepare checks those hashes again, parses the columns into numbers and categories, sorts each year by
the row index, checks every year against the row and label counts read from the published files on 2026-10-13, and
writes one parquet file inside a versioned folder with a readme and a manifest that record the source, the hashes,
the counts and a content fingerprint.
"""

import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from forge import paths
from forge.data import digest
from forge.runs import git_state

PAPER = (
    "Dragoi M., Burceanu E., Haller E., Manolache A., Brad F. AnoShift: A Distribution Shift Benchmark for "
    "Unsupervised Anomaly Detection. NeurIPS 2022 Datasets and Benchmarks Track. arXiv:2206.15476."
)
KYOTO = (
    "Song J., Takakura H., Okabe Y., Eto M., Inoue D., Nakao K. Statistical Analysis of Honeypot Data and Building of "
    "Kyoto 2006+ Dataset for NIDS Evaluation. BADGERS 2011, 29-36. doi:10.1145/1978672.1978676."
)
REPO = "bit-ml/AnoShift"
COMMIT = "096afe36c4401baf85c9bc1f18cd45f9f10df4ef"
URL = "https://storage.googleapis.com/bitdefender_ml_artifacts/anoshift/Kyoto-2016_AnoShift.tar"
# size and last modified time of the tar as Google storage reported them on 2026-10-12
TAR_BYTES = 8099829760
TAR_MODIFIED = "Mon, 11 May 2026 14:23:05 GMT"
# sha256 of the whole tar from the first download, on Colab on 2026-10-12, nothing published to check it against
TAR_SHA256 = "a12485e086ed657509e85dd100c28c5b2b287054b431c9105622a49709ca2920"
SOURCE_FILE = "source.json"

# member size in the tar headers, and rows and label counts read from each test file through its footer and label
# column on 2026-10-13, without downloading the tar
EXPECTED = {
    2006: {"bytes": 5637992, "rows": 413790, "known": 379212, "unknown": 9582, "normal": 24996},
    2007: {"bytes": 33249512, "rows": 2668138, "known": 1618244, "unknown": 31392, "normal": 1018502},
    2008: {"bytes": 58247032, "rows": 4252687, "known": 1443692, "unknown": 944, "normal": 2808051},
    2009: {"bytes": 119775122, "rows": 8527121, "known": 3498147, "unknown": 133944, "normal": 4895030},
    2010: {"bytes": 217896123, "rows": 17033113, "known": 13057517, "unknown": 105576, "normal": 3870020},
    2011: {"bytes": 107597197, "rows": 8549802, "known": 7666277, "unknown": 140681, "normal": 742844},
    2012: {"bytes": 66076787, "rows": 5046105, "known": 4101331, "unknown": 29311, "normal": 915463},
    2013: {"bytes": 128527019, "rows": 9393462, "known": 8741826, "unknown": 6605, "normal": 645031},
    2014: {"bytes": 154685921, "rows": 11053352, "known": 10542061, "unknown": 2009, "normal": 509282},
    2015: {"bytes": 176706090, "rows": 13671940, "known": 13013544, "unknown": 277, "normal": 658119},
}
# rows of the whole year in train and test file together, from the same reading, for the readme
YEAR_ROWS = {
    2006: 4137899, 2007: 26681387, 2008: 42526866, 2009: 85271354, 2010: 170331138,
    2011: 85498019, 2012: 50461062, 2013: 93934617, 2014: 110533529, 2015: 136719391,
}
# the periods of the AnoShift protocol
PERIODS = {"iid": (2006, 2010), "near": (2011, 2013), "far": (2014, 2015)}

# what each processed version changed, written into its readme
VERSIONS = {"v1": "First version."}

# label codes of column 18 and the attack type each one names, in the order of the categories
LABEL_CODES = {"1": "benign", "-1": "known", "-2": "unknown"}
ATTACK_TYPES = list(LABEL_CODES.values())
# duration and byte counts sit in bin i when they fall in (1.1^i - 1, 1.1^(i+1) - 1], bins 0 to 231
N_BINS = 232
INDEX_COLUMN = "__index_level_0__"
RAW_COLUMNS = [str(c) for c in range(14)] + ["18", INDEX_COLUMN]

# processed column, AnoShift column, how it is parsed, Kyoto 2006+ field and meaning, in table order
FEATURES = [
    ("duration_bin", "0", "bins", "Duration", "log bin of the session length in seconds"),
    ("service", "1", "category", "Service", "service type of the session"),
    ("src_bytes_bin", "2", "bins", "Source bytes", "log bin of the bytes sent by the source"),
    ("dst_bytes_bin", "3", "bins", "Destination bytes", "log bin of the bytes sent by the destination"),
    ("count", "4", "count", "Count", "sessions with the same source and destination address in the past 2 s"),
    ("same_srv_rate", "5", "rate", "Same_srv_rate", "share of those sessions to the same service"),
    ("serror_rate", "6", "rate", "Serror_rate", "share of those sessions with SYN errors"),
    ("srv_serror_rate", "7", "rate", "Srv_serror_rate",
     "share of SYN errors among the sessions to the same service in the past 2 s"),
    ("dst_host_count", "8", "count", "Dst_host_count",
     "sessions from the same source among the past 100 to the same destination"),
    ("dst_host_srv_count", "9", "count", "Dst_host_srv_count",
     "sessions to the same service among the past 100 to the same destination"),
    ("dst_host_same_src_port_rate", "10", "rate", "Dst_host_same_src_port_rate",
     "share of the Dst_host_count sessions from the same source port"),
    ("dst_host_serror_rate", "11", "rate", "Dst_host_serror_rate",
     "share of the Dst_host_count sessions with SYN errors"),
    ("dst_host_srv_serror_rate", "12", "rate", "Dst_host_srv_serror_rate",
     "share of the Dst_host_srv_count sessions with SYN errors"),
    ("flag", "13", "category", "Flag", "state of the session when its record was written"),
]
KINDS = {
    "bins": pa.int16(),
    "category": pa.dictionary(pa.int16(), pa.string()),
    "count": pa.int16(),
    "rate": pa.float32(),
}
SCHEMA = pa.schema(
    [("year", pa.int16()), ("order", pa.int32()), ("label", pa.int8()),
     ("attack_type", pa.dictionary(pa.int8(), pa.string()))]
    + [(name, KINDS[kind]) for name, _, kind, _, _ in FEATURES]
)
# source and meaning of the columns that are not Kyoto fields, written into the readme
COLUMN_NOTES = {
    "year": ("file name", "time", "year of the test file"),
    "order": (INDEX_COLUMN, "time", "row position of the session in its year over train and test file, from 0"),
    "label": ("18", "label", "1 attack, known or unknown type, 0 normal"),
    "attack_type": ("18", "label", "benign, known or unknown, from codes 1, -1 and -2"),
}

ROW_GROUP_ROWS = 1_000_000
# bytes read from the network and written to disk at a time
BLOCK = 1 << 20


def member_name(year: int) -> str:
    return f"Kyoto-2016_AnoShift/full/{year}_full_valid.parquet"


def file_name(year: int) -> str:
    return Path(member_name(year)).name


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


class _Hashing(io.RawIOBase):
    # passes a stream through and hashes and counts every byte read from it
    def __init__(self, raw):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        n = self.raw.readinto(b)
        if n:
            self.h.update(memoryview(b)[:n])
            self.n += n
        return n


def download(cfg: dict, log=print, expected: dict | None = None, url: str = URL, tar_bytes: int = TAR_BYTES,
             tar_sha256: str | None = TAR_SHA256) -> Path:
    # a test file gets its final name only once complete, and source.json is written only after the whole tar has
    # passed through the hash, so a raw folder with source.json holds checked copies
    expected = EXPECTED if expected is None else expected
    members = {member_name(year): e["bytes"] for year, e in expected.items()}
    folder = raw_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / SOURCE_FILE
    if source.exists() and all((folder / Path(m).name).exists() and (folder / Path(m).name).stat().st_size == size
                               for m, size in members.items()):
        log(f"found the {len(members)} test files and {SOURCE_FILE} in {folder}")
        return folder
    log(f"streaming {url}, {tar_bytes} bytes, keeping {len(members)} test files")
    found = {}
    with urllib.request.urlopen(url, timeout=120) as response:
        modified = response.headers.get("Last-Modified")
        stream = _Hashing(response)
        reader = io.BufferedReader(stream, buffer_size=BLOCK)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if member.name not in members:
                    continue
                if not member.isfile() or member.size != members[member.name]:
                    raise ValueError(f"{member.name} is {member.size} bytes in the tar, "
                                     f"{members[member.name]} expected")
                # files are written under names from the expected list only, never under a path taken from the tar
                target = folder / Path(member.name).name
                part = target.with_name(f"{target.name}.part")
                h = hashlib.sha256()
                with tar.extractfile(member) as src, open(part, "wb") as dst:
                    while block := src.read(BLOCK):
                        dst.write(block)
                        h.update(block)
                part.replace(target)
                found[member.name] = {"file": target.name, "bytes": member.size, "sha256": h.hexdigest()}
                log(f"kept {target.name}, {member.size} bytes")
        # the end of the archive after the last member, so the hash covers every byte of the tar
        while reader.read(BLOCK):
            pass
    missing = sorted(set(members) - set(found))
    if missing:
        raise ValueError(f"the tar holds no {', '.join(missing)}")
    if stream.n != tar_bytes:
        raise ValueError(f"the tar arrived with {stream.n} bytes where {tar_bytes} were reported")
    if tar_sha256 is not None and stream.h.hexdigest() != tar_sha256:
        raise ValueError(f"the tar arrived with sha256 {stream.h.hexdigest()}, the first download gave {tar_sha256}")
    record = {
        "url": url,
        "bytes": stream.n,
        "sha256": stream.h.hexdigest(),
        "last_modified": modified,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "members": found,
    }
    source.write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"read the whole tar, {stream.n} bytes, sha256 {record['sha256']}")
    return folder


def verify_raw(folder: Path, expected: dict, log=print, tar_sha256: str | None = TAR_SHA256) -> dict:
    source = folder / SOURCE_FILE
    if not source.exists():
        raise FileNotFoundError(f"{source} is missing, download writes it after hashing the tar")
    record = json.loads(source.read_text(encoding="utf-8"))
    if tar_sha256 is not None and record["sha256"] != tar_sha256:
        raise ValueError(f"{source} records a tar with sha256 {record['sha256']}, the first download gave {tar_sha256}")
    for year, e in expected.items():
        path = folder / file_name(year)
        kept = record["members"].get(member_name(year))
        if not path.exists() or kept is None:
            raise FileNotFoundError(f"{path} is missing, download fetches it")
        if path.stat().st_size != e["bytes"] or _sha256(path) != kept["sha256"]:
            raise ValueError(f"{path} differs from the size or sha256 recorded when it was downloaded")
    log(f"checked size and sha256 of {len(expected)} test files against {SOURCE_FILE}")
    return record


def _first_bad(values: pa.Array, bad: pa.Array) -> str:
    return str(values.filter(bad)[0].as_py())


def parse_bins(values: pa.Array, column: str) -> pa.Array:
    # c<column><i> for bin i, nan where the value fell outside the bins, which becomes -1
    prefix = f"c{column}"
    nan = pc.equal(values, "nan")
    bad = pc.invert(pc.or_(nan, pc.match_substring_regex(values, f"^{prefix}[0-9]+$")))
    if pc.any(bad).as_py():
        raise ValueError(f"column {column} holds {_first_bad(values, bad)}, expected {prefix} and a bin number")
    bins = pc.cast(pc.if_else(nan, pa.scalar("-1"), pc.utf8_slice_codeunits(values, len(prefix))), pa.int16())
    if pc.max(bins).as_py() >= N_BINS:
        raise ValueError(f"column {column} holds bin {pc.max(bins).as_py()}, the bins end at {N_BINS - 1}")
    return bins


def parse_number(values: pa.Array, column: str, kind: str) -> pa.Array:
    # counts are whole numbers from 0 to 100 and rates shares from 0 to 1, as written in the published files
    pattern, high = ("^[0-9]+$", 100) if kind == "count" else ("^[0-9]\\.[0-9]+$", 1)
    bad = pc.invert(pc.match_substring_regex(values, pattern))
    if pc.any(bad).as_py():
        raise ValueError(f"column {column} holds {_first_bad(values, bad)}, expected a {kind}")
    numbers = pc.cast(values, KINDS[kind])
    if pc.max(numbers).as_py() > high:
        raise ValueError(f"column {column} holds {pc.max(numbers).as_py()}, above {high}")
    return numbers


def parse_labels(values: pa.Array) -> tuple[pa.Array, pa.Array]:
    codes = pc.index_in(values, value_set=pa.array(list(LABEL_CODES)))
    if codes.null_count:
        raise ValueError(f"column 18 holds {_first_bad(values, pc.is_null(codes))}, expected one of "
                         f"{', '.join(LABEL_CODES)}")
    attack_type = pa.DictionaryArray.from_arrays(pc.cast(codes, pa.int8()), pa.array(ATTACK_TYPES))
    return pc.cast(pc.not_equal(values, "1"), pa.int8()), attack_type


def read_year(path: Path, year: int) -> pa.Table:
    # one test file as a table in SCHEMA, sorted by the row position of each session in its year
    raw = pq.read_table(path)
    if sorted(raw.schema.names) != sorted(RAW_COLUMNS):
        raise ValueError(f"{path.name} has columns {raw.schema.names}, expected {RAW_COLUMNS}")
    order = raw.column(INDEX_COLUMN).combine_chunks()
    if pc.min(order).as_py() < 0 or pc.max(order).as_py() >= 2**31:
        raise ValueError(f"{path.name} holds row positions outside 0 to 2^31 - 1")
    if pc.count_distinct(order).as_py() != len(order):
        raise ValueError(f"{path.name} holds a row position twice")
    label, attack_type = parse_labels(raw.column("18").combine_chunks())
    columns = {
        "year": pa.array(np.full(raw.num_rows, year, dtype=np.int16)),
        "order": pc.cast(order, pa.int32()),
        "label": label,
        "attack_type": attack_type,
    }
    for name, source, kind, _, _ in FEATURES:
        values = raw.column(source).combine_chunks()
        if kind == "bins":
            columns[name] = parse_bins(values, source)
        elif kind == "category":
            columns[name] = values.dictionary_encode().cast(KINDS[kind])
        else:
            columns[name] = parse_number(values, source, kind)
    table = pa.table(columns, schema=SCHEMA)
    return table.take(pc.sort_indices(table.column("order")))


def _summary(flows: pd.DataFrame, year: int) -> dict:
    types = flows["attack_type"].value_counts()
    return {
        "year": year,
        "file": file_name(year),
        "flows": int(len(flows)),
        "attacks": int(flows["label"].sum()),
        "known": int(types.get("known", 0)),
        "unknown": int(types.get("unknown", 0)),
        "normal": int(types.get("benign", 0)),
        "order_min": int(flows["order"].iloc[0]),
        "order_max": int(flows["order"].iloc[-1]),
        "services": int(flows["service"].nunique()),
        "fingerprint": digest.table_digest(flows, SCHEMA),
    }


def prepare(cfg: dict, log=print, expected: dict | None = None, tar_sha256: str | None = TAR_SHA256) -> Path:
    expected = EXPECTED if expected is None else expected
    d = cfg["dataset"]
    version = d["processed_version"]
    if version not in VERSIONS:
        raise ValueError(f"VERSIONS has no description of {version}, the readme of a version says what it changed")
    out = processed_dir(cfg)
    if out.exists():
        raise FileExistsError(f"{out} exists and a processed version is never overwritten, name a new version")
    source = raw_dir(cfg)
    record = verify_raw(source, expected, log, tar_sha256)

    # a version folder appears under its final name only once complete, a partial one is left by an interrupted run
    partial = out.with_name(f"{out.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    summaries = []
    with pq.ParquetWriter(partial / "flows.parquet", SCHEMA, compression="zstd") as writer:
        for year in sorted(expected):
            table = read_year(source / file_name(year), year)
            s = _summary(table.to_pandas(), year)
            e = expected[year]
            seen = {k: s["flows"] if k == "rows" else s[k] for k in ("rows", "known", "unknown", "normal")}
            if seen != {k: e[k] for k in seen}:
                raise ValueError(f"{s['file']} holds {seen}, while the published file held "
                                 f"{ {k: e[k] for k in seen} } when it was read on 2026-10-13")
            writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
            summaries.append(s)
            log(f"read {s['file']}, {s['flows']} sessions, {s['attacks']} attacks, {s['unknown']} of unknown type")
            del table

    manifest = {
        "dataset": d["name"],
        "version": version,
        "changes": VERSIONS[version],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {**git_state(), "python": sys.version.split()[0], "pandas": pd.__version__, "pyarrow": pa.__version__},
        "source": {
            "paper": PAPER,
            "kyoto": KYOTO,
            "repository": f"https://github.com/{REPO}",
            "commit": COMMIT,
            "tar": {k: record[k] for k in ("url", "bytes", "sha256", "last_modified", "downloaded_utc")},
            "members": record["members"],
        },
        "rows": sum(s["flows"] for s in summaries),
        "attacks": sum(s["attacks"] for s in summaries),
        "columns": {f.name: str(f.type) for f in SCHEMA},
        "years": summaries,
        "fingerprint": digest.combine((str(s["year"]), s["fingerprint"]) for s in summaries),
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (partial / "readme.md").write_text(readme(manifest), encoding="utf-8")
    partial.rename(out)
    log(f"wrote {manifest['rows']} sessions to {out}, fingerprint {manifest['fingerprint']}")
    return out


def read_manifest(cfg: dict) -> dict:
    return json.loads((processed_dir(cfg) / "manifest.json").read_text(encoding="utf-8"))


def load(cfg: dict, columns: list[str] | None = None, years: list[int] | None = None) -> pd.DataFrame:
    # the processed table in the order it was written, by year and then by row position in the year, only the given
    # years when years is set
    filters = None if years is None else [("year", "in", [int(y) for y in years])]
    flows = pq.read_table(processed_dir(cfg) / "flows.parquet", columns=columns, filters=filters).to_pandas()
    if "attack_type" in flows.columns:
        flows["attack_type"] = flows["attack_type"].cat.set_categories(ATTACK_TYPES)
    return flows


def fingerprint(flows: pd.DataFrame) -> str:
    # recomputes the fingerprint of a full table read back with load, one digest per year in row order
    return digest.fingerprint(flows, "year", SCHEMA)


def period(year: int) -> str:
    return next(name for name, (first, last) in PERIODS.items() if first <= year <= last)


def readme(m: dict) -> str:
    code, src = m["code"], m["source"]
    tar = src["tar"]
    dirty = ", which had uncommitted changes" if code["dirty"] else ""
    lines = [
        f"# AnoShift, processed {m['version']}",
        "",
        f"Written by `forge.data.anoshift.prepare` on {m['created_utc']} from forge-nids commit "
        f"`{code['commit'] or 'unknown'}`{dirty}. Python {code['python']}, pandas {code['pandas']}, "
        f"pyarrow {code['pyarrow']}. {m['changes']}",
        "",
        f"`flows.parquet` holds {m['rows']} sessions, {m['attacks']} of them attacks, the test files of AnoShift for "
        f"the {len(m['years'])} years from {m['years'][0]['year']} to {m['years'][-1]['year']}, sorted by `year` and "
        "`order`. `manifest.json` holds the same facts in machine readable form.",
        "",
        "## Source",
        "",
        f"{src['paper']} The sessions come from Kyoto 2006+, {src['kyoto']} Code at {src['repository']}, commit "
        f"`{src['commit']}`, BSD 3-Clause.",
        "",
        f"The data is one tar, {tar['url']}, {tar['bytes']} bytes, last modified {tar['last_modified']}, sha256 "
        f"`{tar['sha256']}`, downloaded {tar['downloaded_utc']}. No hash is published with it, so the sha256 is the "
        "one computed while the tar was streamed, and the one of the first download is pinned in "
        "`forge.data.anoshift`. Only the test file of each year is kept.",
        "",
        "| member | bytes | sha256 |",
        "|---|---|---|",
    ]
    for name, f in src["members"].items():
        lines.append(f"| {name} | {f['bytes']} | `{f['sha256']}` |")
    lines += [
        "",
        "## Steps",
        "",
        "1. The tar is streamed once from the address above. Every byte passes through the sha256, and only the test "
        "files are written to disk, each under a name from a fixed list and with its own sha256.",
        "2. The test file of a year holds a random 10% of each of its label classes, the train file the rest, so the "
        "class shares of a test file are those of the whole year. The train files are not used.",
        "3. Duration and the byte counts arrive as bin labels, `c0` to `c3` followed by the bin number, which is kept "
        "as an integer. A label `nan` would mark a value outside the bins and become -1. Counts and rates are parsed "
        "as numbers, service and flag kept as categories. Processing stops on any value outside these formats.",
        "4. Column 18 gives `label` and `attack_type`, and processing stops on a code other than 1, -1 and -2.",
        "5. `order` is the pandas index of the published file, the row position of the session in its year over the "
        "train and test file, unique within the year, which was checked. Rows are sorted by `year` and `order`.",
        "6. Every year is checked against the rows and label counts read from the published test files on "
        "2026-10-13. No row is filtered, deduplicated or changed.",
        "",
        "## Columns",
        "",
        "| column | type | from | role | meaning |",
        "|---|---|---|---|---|",
    ]
    features = {name: (source, field, meaning) for name, source, _, field, meaning in FEATURES}
    for name, dtype in m["columns"].items():
        dtype = "category" if dtype.startswith("dictionary") else dtype
        if name in COLUMN_NOTES:
            origin, role, meaning = COLUMN_NOTES[name]
        else:
            source, field, meaning = features[name]
            origin, role = f"{source}, Kyoto field {field}", "feature"
        lines.append(f"| `{name}` | {dtype} | {origin} | {role} | {meaning} |")
    lines += [
        "",
        f"A bin i holds values in (1.1^i - 1, 1.1^(i+1) - 1], i from 0 to {N_BINS - 1}. Kyoto fields 14 to 16, the "
        "detector alerts, and 18 to 23, addresses, ports, start time and protocol, are not in the published files. "
        "Their column 18 holds the values of field 17, the label.",
        "",
        "## Years",
        "",
        "The period is the one of the AnoShift protocol. Sessions of the whole year count train and test file "
        "together. The fingerprint column is the start of each year's digest, see Fingerprint.",
        "",
        "| year | period | sessions of the year | sessions here | attacks | known | unknown | normal | attack share | "
        "order from | order to | fingerprint |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for y in m["years"]:
        lines.append(f"| {y['year']} | {period(y['year'])} | {YEAR_ROWS.get(y['year'], '')} | {y['flows']} | "
                     f"{y['attacks']} | {y['known']} | {y['unknown']} | {y['normal']} | "
                     f"{y['attacks'] / y['flows']:.4f} | "
                     f"{y['order_min']} | {y['order_max']} | `{y['fingerprint'][:12]}` |")
    lines += [
        "",
        "## Differences from the paper",
        "",
        "- Table 7 of the paper gives 806M sessions, which matches the 806095262 rows of the train and test files of "
        "all years together.",
        "- Section 3.1 gives 89.5% anomalies, while the published files hold 80.04% attacks over all years.",
        "- Appendix B says column 14 holds the full timestamp, while the published files have no column 14 and no "
        "time at all.",
        "- `data_processor/parse_kyoto_logbins.py` in the repository keeps Kyoto fields 0 to 17 with a timestamp and "
        "the protocol and labels by field 14, so the published files come from a different version of it.",
        "",
        "## Fingerprint",
        "",
        f"`{m['fingerprint']}`",
        "",
        "A sha256 over the values of every column in row order, one digest per year, independent of how the Parquet "
        "file is encoded. `forge.data.anoshift.fingerprint(forge.data.anoshift.load(cfg))` recomputes it, so a table "
        "built again from the same files can be checked against this one.",
        "",
    ]
    return "\n".join(lines)
