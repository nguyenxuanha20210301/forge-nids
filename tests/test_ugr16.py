import io
import json
import tarfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from forge.data import ugr16 as U

from conftest import UGR16_WEEKS as WEEKS
from conftest import ugr16_line as line
from conftest import ugr16_week_rows as week_rows
from conftest import write_ugr16_mirror as write_mirror


def archive(target, lines):
    # one csv member of the given lines, written as bytes so that a line can hold anything
    data = b"\n".join(lines) + b"\n"
    with tarfile.open(target, "w:gz") as tar:
        info = tarfile.TarInfo("uniq/august.week1.csv.uniqblacklistremoved")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return target


def test_family_name_maps_known_labels_and_keeps_others():
    assert U.family_name("background") == "benign" and U.family_name("anomaly-sshscan") == "anomaly_sshscan"
    assert U.family_name("anomaly-newkind") == "anomaly_newkind" and U.family_name("") == "unlabelled"
    assert U.FAMILY_ORDER[0] == "benign" and set(U.INJECTED_FAMILIES) < set(U.ATTACK_FAMILIES)
    assert U.official_path("test/august/week5/august_week5_csv.tar.gz") == \
        "download/attack/august/week5/august_week5_csv.tar.gz"
    assert U.SCHEMA.names == list(U.COLUMN_NOTES)


def test_weeks_of_keeps_the_time_order_and_rejects_unknown_names():
    cfg = {"dataset": {"weeks": ["august_week2", "july_week5"]}}
    assert U.weeks_of(cfg, U.FILES) == ["july_week5", "august_week2"]
    with pytest.raises(ValueError, match="unknown weeks"):
        U.weeks_of({"dataset": {"weeks": ["march_week3"]}}, U.FILES)
    with pytest.raises(ValueError, match="sample_every"):
        U.sample_every({"dataset": {"sample_every": 0}})


def test_prepare_keeps_every_kth_row_in_time_order(ugr16_mirror):
    m = ugr16_mirror
    out = U.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=lambda s: None)
    flows = U.load(m.cfg)
    manifest = U.read_manifest(m.cfg)
    assert manifest["sample_every"] == 3 and manifest["rows_in_weeks"] == 80
    assert list(flows["week"].cat.categories) == WEEKS
    for week, rows in m.truth.items():
        mine = flows[flows["week"] == week]
        # rows 1, 4, 7 and on of the 40 parsed lines, the blank and the short line not counted
        expected = [rows[i] for i in range(0, 40, 3)]
        assert sorted(mine["row_id"].tolist()) == [i + 1 for i in range(0, 40, 3)]
        assert mine["te"].is_monotonic_increasing
        by_id = mine.set_index("row_id").sort_index()
        assert by_id["te"].tolist() == [pd.Timestamp(r["te"]) for r in expected]
        assert by_id["family"].astype(str).tolist() == [U.family_name(r["label"]) for r in expected]
        assert by_id["label"].tolist() == [int(r["label"] != "background") for r in expected]
        assert by_id["dp"].tolist() == [r["dp"] for r in expected]
        assert by_id["proto"].astype(str).tolist() == [r["pr"] for r in expected]
        s = next(w for w in manifest["weeks"] if w["week"] == week)
        assert s["flows_in_week"] == 40 and s["skipped_lines"] == 1 and s["lines"] == 41
        assert s["flows"] == 14 and sum(s["labels_in_week"].values()) == 40
        assert s["labels_in_week"] == {U.family_name(k): v for k, v in
                                       pd.Series([r["label"] for r in rows]).value_counts().items()}
        assert s["unknown_labels"] == ["anomaly-newkind"]
        assert s["dropped"] == 0 and s["dropped_examples"] == []
    assert manifest["dropped"]["rows"] == 0
    assert next(w for w in manifest["weeks"] if w["week"] == "july_week5")["in_order"]
    assert not next(w for w in manifest["weeks"] if w["week"] == "august_week1")["in_order"]
    # ties keep the file order, and the swapped pair is sorted back
    august = flows[flows["week"] == "august_week1"]
    ties = august.groupby("te")["row_id"].apply(lambda s: s.is_monotonic_increasing)
    assert ties.all()
    assert sorted(flows["tcp_flags"].unique().tolist()) <= [0, 2, 16, 27, 194]
    assert "anomaly_newkind" in flows["family"].cat.categories
    assert list(flows["family"].cat.categories[:len(U.FAMILY_ORDER)]) == U.FAMILY_ORDER or \
        flows["family"].cat.categories[0] == "benign"
    assert U.fingerprint(flows) == manifest["fingerprint"]
    # the weeks lie days apart, so the boundary merges nothing and the whole table is in te order
    assert flows["te"].is_monotonic_increasing
    kept = {w: [rows[i] for i in range(0, 40, 3)] for w, rows in m.truth.items()}
    gap = max(r["te"] for r in kept["july_week5"]) - min(r["te"] for r in kept["august_week1"])
    assert manifest["boundaries"] == [{"weeks": WEEKS, "overlap_seconds": gap.total_seconds(), "rows": [0, 0]}]
    assert (out / "readme.md").exists() and "anomaly-newkind" in (out / "readme.md").read_text()
    # the archives were deleted from the scratch space and no part is left
    assert not list(m.scratch.glob("*.tar.gz")) and not (out / "parts").exists()


def shifted(week, seed, start):
    # the rows of a week moved so that the week starts at the given time
    rows = week_rows(week, seed, swap=False)
    shift = rows[0]["te"] - start
    for r in rows:
        r["te"] = r["te"] - shift
    return rows


def test_prepare_merges_the_rows_of_overlapping_weeks(ugr16_mirror, tmp_path):
    # the august week starts 15 s into the july week, as the archives overlap at every boundary by up to a minute
    m = ugr16_mirror
    july = week_rows("july_week5", 1, swap=False)
    august = shifted("august_week1", 2, july[0]["te"] + timedelta(seconds=15))
    files, truth = write_mirror(tmp_path / "mirror3", {"july_week5": july, "august_week1": august})
    out = U.prepare(m.cfg, files=files, url=f"{(tmp_path / 'mirror3').as_uri()}/{{path}}", scratch=m.scratch,
                    log=lambda s: None)
    flows = U.load(m.cfg)
    manifest = U.read_manifest(m.cfg)
    kept = {w: [(pd.Timestamp(r["te"]), i + 1) for i, r in enumerate(rows) if i % 3 == 0] for w, rows in truth.items()}
    expected = sorted((te, WEEKS.index(w), row_id) for w, rows in kept.items() for te, row_id in rows)
    mine = list(zip(flows["te"].tolist(), flows["week"].astype(str).map(WEEKS.index).tolist(),
                    flows["row_id"].tolist()))
    assert mine == expected and flows["te"].is_monotonic_increasing
    first_august = min(te for te, _ in kept["august_week1"])
    last_july = max(te for te, _ in kept["july_week5"])
    b = manifest["boundaries"][0]
    assert b["weeks"] == WEEKS and b["overlap_seconds"] == (last_july - first_august).total_seconds() > 0
    assert b["rows"] == [sum(te >= first_august for te, _ in kept["july_week5"]),
                         sum(te <= last_july for te, _ in kept["august_week1"])] and min(b["rows"]) > 0
    assert len(U.load(m.cfg, weeks=["august_week1"])) == 14 and manifest["rows"] == 28
    assert U.fingerprint(flows) == manifest["fingerprint"]
    readme = (out / "readme.md").read_text()
    assert "## Boundaries" in readme and f"| july_week5 | august_week1 | {b['overlap_seconds']:.0f} |" in readme


def test_prepare_rejects_weeks_out_of_te_order(ugr16_mirror, tmp_path):
    m = ugr16_mirror
    july = week_rows("july_week5", 1, swap=False)
    august = shifted("august_week1", 2, july[0]["te"] - timedelta(days=1))
    files, _ = write_mirror(tmp_path / "mirror4", {"july_week5": july, "august_week1": august})
    with pytest.raises(ValueError, match="not in te order"):
        U.prepare(m.cfg, files=files, url=f"{(tmp_path / 'mirror4').as_uri()}/{{path}}", scratch=m.scratch,
                  log=lambda s: None)


def test_prepare_rejects_a_week_that_overlaps_the_week_after_next(ugr16_mirror, tmp_path):
    # three weeks a few seconds apart, so the first still runs when the third starts
    m = ugr16_mirror
    july = week_rows("july_week5", 1, swap=False)
    weeks = {"july_week5": july, "august_week1": shifted("august_week1", 2, july[0]["te"] + timedelta(seconds=5)),
             "august_week2": shifted("august_week1", 3, july[0]["te"] + timedelta(seconds=10))}
    files, _ = write_mirror(tmp_path / "mirror5", weeks)
    m.cfg["dataset"]["weeks"] = list(weeks)
    with pytest.raises(ValueError, match="overlap in te although a week lies between them"):
        U.prepare(m.cfg, files=files, url=f"{(tmp_path / 'mirror5').as_uri()}/{{path}}", scratch=m.scratch,
                  log=lambda s: None)


def test_load_filters_weeks(ugr16_mirror):
    m = ugr16_mirror
    U.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=lambda s: None)
    flows = U.load(m.cfg, columns=["te", "week", "label"], weeks=["august_week1"])
    assert flows["week"].astype(str).unique().tolist() == ["august_week1"] and len(flows) == 14


def test_prepare_resumes_from_the_weeks_already_written(ugr16_mirror, monkeypatch):
    m = ugr16_mirror
    calls = []
    original = U.fetch

    def failing(path, url, size, sha256, log=print):
        calls.append(path.name)
        if "august" in path.name:
            raise OSError("the network went away")
        return original(path, url, size, sha256, log)

    monkeypatch.setattr(U, "fetch", failing)
    with pytest.raises(OSError):
        U.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=lambda s: None)
    partial = U.processed_dir(m.cfg).with_name("v1.partial")
    assert (partial / "parts" / "july_week5.parquet").exists() and not U.processed_dir(m.cfg).exists()
    def counting(path, url, size, sha256, log=print):
        calls.append(path.name)
        return original(path, url, size, sha256, log)

    monkeypatch.setattr(U, "fetch", counting)
    messages = []
    U.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=messages.append)
    # the july week was fetched once, in the first run, and found in the second
    assert calls.count("july_week5_csv.tar.gz") == 1 and calls.count("august_week1_csv.tar.gz") == 2
    assert any(s.startswith("found july_week5 in") for s in messages)
    assert U.read_manifest(m.cfg)["rows"] == 28 and not partial.exists()


def test_prepare_never_overwrites_a_version(ugr16_mirror):
    m = ugr16_mirror
    U.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=lambda s: None)
    with pytest.raises(FileExistsError):
        U.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=lambda s: None)


def test_prepare_rejects_a_changed_archive(ugr16_mirror):
    m = ugr16_mirror
    files = dict(m.files)
    path, size, _ = files["july_week5"]
    files["july_week5"] = (path, size, "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        U.prepare(m.cfg, files=files, url=m.url, scratch=m.scratch, log=lambda s: None)
    assert not U.processed_dir(m.cfg).exists() and not list(m.scratch.glob("*.tar.gz"))


def test_download_fetches_nothing_in_scratch_mode_and_the_archives_otherwise(ugr16_mirror):
    m = ugr16_mirror
    messages = []
    folder = U.download(m.cfg, files=m.files, url=m.url, log=messages.append)
    assert not list(folder.glob("*.tar.gz")) if folder.exists() else True
    assert messages[0].startswith("raw_scratch is set")
    cfg = {"dataset": {**m.cfg["dataset"], "raw_scratch": False}}
    folder = U.download(cfg, files=m.files, url=m.url, log=messages.append)
    assert folder == m.storage / "data" / "raw" / "ugr16"
    assert sorted(p.name for p in folder.glob("*.tar.gz")) == ["august_week1_csv.tar.gz", "july_week5_csv.tar.gz"]
    # the second call finds the checked copies
    U.download(cfg, files=m.files, url=m.url, log=messages.append)
    assert sum(s.startswith("found ") for s in messages) == 2


def test_read_week_drops_the_rows_with_a_value_it_cannot_read(tmp_path):
    # a thousand good lines, then a port that runs into the time of the next line as the archives have it, a port out
    # of range, an hour out of range, a duration that is not a number, a byte outside utf-8 in a port, and one in a
    # label, which is a value the reader can keep under its own name
    start = datetime(2016, 8, 1)
    good = [line(start + timedelta(seconds=i)).encode() for i in range(1000)]
    t = start + timedelta(seconds=1000)
    bad = [line(t, dp="56572016-08-03 15:14:34").encode(), line(t, sp=70000).encode(),
           b"2016-08-01 25:00:00" + line(t).encode()[19:], line(t, td="abc").encode(),
           line(t, dp="8X0").encode().replace(b"8X0", b"8\xff0"),
           line(t, label="backgrXund").encode().replace(b"backgrXund", b"backgr\xffund")]
    target = archive(tmp_path / "week.tar.gz", good + bad)
    messages = []
    flows, s = U.read_week(target, "august_week1", WEEKS, 1, log=messages.append)
    assert s["lines"] == 1006 and s["skipped_lines"] == 0 and s["flows_in_week"] == 1006
    assert s["dropped"] == 5 and s["flows"] == 1001 and len(flows) == 1001
    assert [(e["row_id"], e["column"]) for e in s["dropped_examples"]] == \
        [(1001, "dp"), (1002, "sp"), (1003, "te"), (1004, "td"), (1005, "dp")]
    assert [e["value"] for e in s["dropped_examples"]] == \
        ["56572016-08-03 15:14:34", "70000", "2016-08-01 25:00:00", "abc", "8\ufffd0"]
    assert not set(range(1001, 1006)) & set(flows["row_id"]) and 1006 in set(flows["row_id"])
    # the line with the odd byte in its label counts under that label, kept with the replacement character
    assert s["labels_in_week"] == {"benign": 1005, "backgr\ufffdund": 1} and s["unknown_labels"] == ["backgr\ufffdund"]
    assert s["attacks"] == 1
    assert flows.loc[flows["row_id"] == 1006, "family"].astype(str).tolist() == ["backgr\ufffdund"]
    assert flows["dp"].tolist()[:1000] == [80] * 1000 and flows["te"].is_monotonic_increasing
    assert "5 kept rows dropped" in messages[-1] and "line 1001 dp" in messages[-1]


def test_read_week_rejects_a_week_it_cannot_read(tmp_path):
    # every port out of range means the fields are not what the columns say, not a few cut lines
    start = datetime(2016, 8, 1)
    lines = [line(start + timedelta(seconds=i), sp=70000).encode() for i in range(20)]
    target = archive(tmp_path / "week.tar.gz", lines)
    with pytest.raises(ValueError, match="cannot be read"):
        U.read_week(target, "august_week1", WEEKS, 1, log=lambda s: None)


def test_prepare_records_the_dropped_flows(ugr16_mirror, tmp_path, monkeypatch):
    # one cut line at a kept position of the august week, with the share a small fixture needs
    m = ugr16_mirror
    monkeypatch.setattr(U, "MAX_DROPPED", 0.5)
    august = week_rows("august_week1", 2, swap=True)
    august[3]["dp"] = "56572016-08-03 15:14:34"
    files, _ = write_mirror(tmp_path / "mirror2", {"july_week5": week_rows("july_week5", 1, swap=False),
                                                     "august_week1": august})
    out = U.prepare(m.cfg, files=files, url=f"{(tmp_path / 'mirror2').as_uri()}/{{path}}", scratch=m.scratch,
                    log=lambda s: None)
    manifest = U.read_manifest(m.cfg)
    s = next(w for w in manifest["weeks"] if w["week"] == "august_week1")
    assert s["dropped"] == 1 and s["flows"] == 13 and s["flows_in_week"] == 40
    assert s["dropped_examples"] == [{"row_id": 4, "column": "dp", "value": "56572016-08-03 15:14:34"}]
    assert manifest["rows"] == 27 and manifest["dropped"]["rows"] == 1
    flows = U.load(m.cfg)
    assert 4 not in set(flows.loc[flows["week"] == "august_week1", "row_id"])
    assert (flows["week"] == "july_week5").sum() == 14
    readme = (out / "readme.md").read_text()
    assert "- august_week1, line 4, dp `56572016-08-03 15:14:34`" in readme and "| no | 1 | 1 | scan11" in readme


def test_read_week_rejects_a_second_member(tmp_path):
    target = tmp_path / "two.tar.gz"
    with tarfile.open(target, "w:gz") as tar:
        for name in ("uniq/a.csv", "uniq/b.csv"):
            data = line(datetime(2016, 8, 1)).encode() + b"\n"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="one csv was expected"):
        U.read_week(target, "august_week1", WEEKS, 1, log=lambda s: None)


def test_manifest_records_the_mirror_and_the_official_paths(ugr16_mirror):
    m = ugr16_mirror
    U.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=lambda s: None)
    manifest = U.read_manifest(m.cfg)
    assert manifest["source"]["mirror"].endswith(U.MIRROR)
    assert manifest["source"]["files"]["july_week5"]["official_path"] == \
        "download/attack/july/week5/july_week5_csv.tar.gz"
    assert json.dumps(manifest)
