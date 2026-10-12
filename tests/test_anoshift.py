import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from forge.data import anoshift


def quiet(message):
    pass


def download(src, expected=None, tar_bytes=None, tar_sha256=None):
    return anoshift.download(src.cfg, log=quiet, expected=src.expected if expected is None else expected, url=src.url,
                             tar_bytes=src.tar_bytes if tar_bytes is None else tar_bytes,
                             tar_sha256=src.tar_sha256 if tar_sha256 is None else tar_sha256)


def prepare(src, expected=None):
    return anoshift.prepare(src.cfg, log=quiet, expected=src.expected if expected is None else expected,
                            tar_sha256=src.tar_sha256)


def truth(src):
    # the rows of the tiny test files in the order prepare must write them
    frames = [f.assign(year=year).rename_axis("order").reset_index() for year, f in src.frames.items()]
    return pd.concat(frames).sort_values(["year", "order"], ignore_index=True)


def test_download_keeps_the_test_files_and_hashes_the_whole_tar(anoshift_source, monkeypatch):
    # reads far smaller than a tar record give the same files and hashes
    monkeypatch.setattr(anoshift, "BLOCK", 512)
    folder = download(anoshift_source)
    expected = anoshift_source.expected
    assert sorted(p.name for p in folder.iterdir()) == sorted([anoshift.file_name(y) for y in expected]
                                                              + [anoshift.SOURCE_FILE])
    record = json.loads((folder / anoshift.SOURCE_FILE).read_text())
    assert record["sha256"] == hashlib.sha256(anoshift_source.tar.read_bytes()).hexdigest()
    assert record["bytes"] == anoshift_source.tar_bytes
    for year in expected:
        kept = record["members"][anoshift.member_name(year)]
        assert kept["sha256"] == hashlib.sha256((folder / anoshift.file_name(year)).read_bytes()).hexdigest()
    # checked copies are found again, so the address is not opened a second time
    anoshift.download(anoshift_source.cfg, log=quiet, expected=expected, url="file:///nowhere", tar_bytes=0,
                      tar_sha256="0" * 64)


def test_download_hashes_bytes_after_the_end_of_the_archive(anoshift_source, monkeypatch):
    # tarfile stops reading at the end of archive marker, so the sha256 covers the file as served only if download
    # reads on to the last byte. small reads keep the buffer from fetching the trailing bytes by chance
    monkeypatch.setattr(anoshift, "BLOCK", 512)
    with open(anoshift_source.tar, "ab") as f:
        f.write(b"trailing bytes after the archive" * 100)
    data = anoshift_source.tar.read_bytes()
    folder = download(anoshift_source, tar_bytes=len(data), tar_sha256=hashlib.sha256(data).hexdigest())
    record = json.loads((folder / anoshift.SOURCE_FILE).read_text())
    assert record["bytes"] == len(data) and record["sha256"] == hashlib.sha256(data).hexdigest()


def test_download_rejects_a_tar_or_a_member_of_another_size(anoshift_source):
    with pytest.raises(ValueError):
        download(anoshift_source, tar_bytes=anoshift_source.tar_bytes + 1)
    assert not (anoshift.raw_dir(anoshift_source.cfg) / anoshift.SOURCE_FILE).exists()
    wrong = {**anoshift_source.expected}
    wrong[2011] = {**wrong[2011], "bytes": wrong[2011]["bytes"] + 1}
    with pytest.raises(ValueError):
        download(anoshift_source, expected=wrong)


def test_download_and_prepare_stop_on_a_tar_other_than_the_pinned_one(anoshift_source):
    with pytest.raises(ValueError, match="first download"):
        download(anoshift_source, tar_sha256="0" * 64)
    assert not (anoshift.raw_dir(anoshift_source.cfg) / anoshift.SOURCE_FILE).exists()
    download(anoshift_source)
    with pytest.raises(ValueError, match="first download"):
        anoshift.prepare(anoshift_source.cfg, log=quiet, expected=anoshift_source.expected, tar_sha256="0" * 64)
    assert anoshift.TAR_SHA256 == "a12485e086ed657509e85dd100c28c5b2b287054b431c9105622a49709ca2920"


def test_prepare_sorts_each_year_and_parses_every_column(anoshift_source):
    download(anoshift_source)
    cfg = anoshift_source.cfg
    prepare(anoshift_source)
    flows, rows = anoshift.load(cfg), truth(anoshift_source)
    assert flows["year"].tolist() == rows["year"].tolist()
    assert flows["order"].tolist() == rows["order"].tolist()
    assert flows["label"].tolist() == (rows["18"] != "1").astype(int).tolist()
    assert flows["attack_type"].astype(str).tolist() == rows["18"].map(anoshift.LABEL_CODES).tolist()
    assert flows["duration_bin"].tolist() == rows["0"].str[2:].astype(int).tolist()
    assert flows["src_bytes_bin"].tolist() == [-1 if v == "nan" else int(v[2:]) for v in rows["2"]]
    assert flows["count"].tolist() == rows["4"].astype(int).tolist()
    assert np.allclose(flows["same_srv_rate"], rows["5"].astype(float))
    assert flows["service"].astype(str).tolist() == rows["1"].tolist()
    assert flows["flag"].astype(str).tolist() == rows["13"].tolist()

    manifest = anoshift.read_manifest(cfg)
    assert manifest["rows"] == len(rows) and manifest["attacks"] == int((rows["18"] != "1").sum())
    assert [y["year"] for y in manifest["years"]] == sorted(anoshift_source.expected)
    assert anoshift.fingerprint(flows) == manifest["fingerprint"]
    assert manifest["source"]["tar"]["sha256"] == hashlib.sha256(anoshift_source.tar.read_bytes()).hexdigest()
    readme = (anoshift.processed_dir(cfg) / "readme.md").read_text()
    assert "## Years" in readme and manifest["fingerprint"] in readme
    with pytest.raises(FileExistsError):
        prepare(anoshift_source)


def test_prepare_stops_on_counts_other_than_the_published_ones(anoshift_source):
    download(anoshift_source)
    wrong = {**anoshift_source.expected}
    wrong[2011] = {**wrong[2011], "unknown": wrong[2011]["unknown"] + 1}
    with pytest.raises(ValueError, match="2011_full_valid"):
        prepare(anoshift_source, expected=wrong)
    assert not anoshift.processed_dir(anoshift_source.cfg).exists()


def test_prepare_stops_when_a_raw_file_changed(anoshift_source):
    folder = download(anoshift_source)
    path = folder / anoshift.file_name(2014)
    data = bytearray(path.read_bytes())
    data[100] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        prepare(anoshift_source)


def test_parsers_take_the_published_formats_and_stop_on_others():
    assert anoshift.parse_bins(pa.array(["c00", "c0231", "nan"]), "0").to_pylist() == [0, 231, -1]
    for bad in (["c10"], ["c0232"], ["c0 5"], ["5"]):
        with pytest.raises(ValueError):
            anoshift.parse_bins(pa.array(bad), "0")
    assert anoshift.parse_number(pa.array(["0", "100"]), "4", "count").to_pylist() == [0, 100]
    assert np.allclose(anoshift.parse_number(pa.array(["0.00", "0.33", "1.00"]), "5", "rate").to_pylist(), [0, 0.33, 1])
    for values, kind in ((["101"], "count"), (["1.5"], "count"), (["1.01"], "rate"), (["nan"], "rate"),
                         (["-0.10"], "rate")):
        with pytest.raises(ValueError):
            anoshift.parse_number(pa.array(values), "4", kind)
    label, attack_type = anoshift.parse_labels(pa.array(["1", "-1", "-2"]))
    assert label.to_pylist() == [0, 1, 1] and attack_type.to_pylist() == ["benign", "known", "unknown"]
    with pytest.raises(ValueError):
        anoshift.parse_labels(pa.array(["1", "2"]))
