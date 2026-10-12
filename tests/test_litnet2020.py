import hashlib

import numpy as np
import pandas as pd
import pytest

from forge.data import litnet2020 as L


def test_column_layout():
    assert len(L.COLUMNS) == 85 and len(set(L.COLUMNS)) == 85
    assert len(L.RULE_COLUMNS) == 19
    assert (L.RULE_COLUMNS[0], L.RULE_COLUMNS[-1]) == ("icmp_dst_ip_b", "udp_src_p_0")
    assert len(L.CONSTANT_COLUMNS) == 31
    assert not set(L.USECOLS) & (set(L.RULE_COLUMNS) | set(L.IDENTIFIER_COLUMNS))
    assert set(L.DTYPES) == set(L.USECOLS)
    assert L.SCHEMA.names == list(L.COLUMN_NOTES)


def test_label_mapping_covers_every_code():
    codes = list(L.FAMILIES)
    attack_a = pd.Series([int(c != "none") for c in codes])
    label, family = L.map_labels(pd.Series(codes), attack_a)
    assert label.tolist() == attack_a.tolist()
    assert list(family) == [L.FAMILIES[c] for c in codes]
    assert list(family.categories) == L.FAMILY_ORDER
    assert L.FAMILY_ORDER[0] == "benign" and set(L.PAPER_TABLE4) == set(L.ATTACK_FAMILIES)


@pytest.mark.parametrize("attack_t, attack_a", [("udp_f", 0), ("none", 1)])
def test_label_and_family_must_agree(attack_t, attack_a):
    with pytest.raises(ValueError, match="disagrees"):
        L.map_labels(pd.Series(["none", attack_t]), pd.Series([0, attack_a]))


def test_unknown_code_stops():
    with pytest.raises(ValueError, match="unknown attack_t"):
        L.map_labels(pd.Series(["none", "tcp_new"]), pd.Series([0, 1]))


def test_label_outside_zero_and_one_stops():
    with pytest.raises(ValueError, match="other than 0 and 1"):
        L.map_labels(pd.Series(["udp_f"]), pd.Series([2]))


@pytest.mark.parametrize("text, value", [
    ("......", 0), ("....S.", 0x02), (".A..S.", 0x12), (".AP.SF", 0x1B), ("UAPRSF", 0x3F),
    ("  0xc2", 0xC2), ("  0x5a", 0x5A),
])
def test_flag_value(text, value):
    assert L.flag_value(text) == value


@pytest.mark.parametrize("text", ["..X...", "S.....", "  0xzz", "  0x 1", "....S", "0xc2  "])
def test_flag_value_rejects_other_strings(text):
    with pytest.raises(ValueError):
        L.flag_value(text)


def test_parse_flags_reads_split_columns():
    rows = ["....S.", "  0xc2", "......", "....S."]
    raw = pd.DataFrame({c: pd.Categorical([r[i] for r in rows]) for i, c in enumerate(L.FLAG_COLUMNS)})
    assert L.parse_flags(raw).tolist() == [2, 194, 0, 2]


def test_parse_times():
    raw = pd.DataFrame({f"te_{u}": [v] for u, v in zip(L.TIME_UNITS, (2020, 1, 5, 23, 59, 59))})
    assert L.parse_times(raw, "te")[0] == np.datetime64("2020-01-05T23:59:59")
    raw["te_month"] = 13
    with pytest.raises(ValueError):
        L.parse_times(raw, "te")


def test_prepare_keeps_every_row_in_time_order(litnet_raw):
    # small chunks so each capture is read in several pieces and joined again
    out = L.prepare(litnet_raw.cfg, files=litnet_raw.files, duplicates=litnet_raw.duplicates, chunksize=4,
                    log=lambda m: None)
    flows = L.load(litnet_raw.cfg)
    truth = pd.DataFrame(litnet_raw.rows)
    assert len(flows) == len(truth)
    assert flows["te"].is_monotonic_increasing
    # b ends before a starts, although a comes first by file name, and the duplicate archive is read zero times
    assert flows["capture"].astype(str).drop_duplicates().tolist() == ["b", "a"]
    # flows that end in the same second keep the order of the source file
    ties = flows.groupby(["capture", "te"], observed=True)["row_id"].apply(lambda s: s.is_monotonic_increasing)
    assert ties.all() and flows.duplicated(["capture", "te"]).any()
    merged = flows.assign(capture=flows["capture"].astype(str)).merge(truth, on=["capture", "row_id"])
    assert len(merged) == len(truth)
    assert (merged["label"] == merged["attack_a"]).all()
    assert (merged["family"].astype(str) == merged["attack_t"].map(L.FAMILIES)).all()
    assert (merged["te_x"] == merged["te_y"]).all()
    assert sorted(flows.loc[flows["capture"] == "b", "tcp_flags"].unique().tolist()) == [0, 2, 194]
    manifest = L.read_manifest(litnet_raw.cfg)
    assert manifest["rows"] == len(truth) and manifest["attacks"] == int(truth["attack_a"].sum())
    assert not manifest["source"]["files"]["C_FLOWS.zip"]["used"]
    assert L.fingerprint(flows) == manifest["fingerprint"]
    assert (out / "readme.md").exists()


def test_fingerprint_changes_with_any_value_or_order(litnet_raw):
    L.prepare(litnet_raw.cfg, files=litnet_raw.files, duplicates=litnet_raw.duplicates, log=lambda m: None)
    flows = L.load(litnet_raw.cfg)
    changed = flows.copy()
    changed.loc[3, "ibyt"] += 1
    swapped = flows.iloc[[1, 0, *range(2, len(flows))]].reset_index(drop=True)
    reference = L.fingerprint(flows)
    assert L.fingerprint(changed) != reference
    assert L.fingerprint(swapped) != reference


def test_prepare_never_overwrites_a_version(litnet_raw):
    L.prepare(litnet_raw.cfg, files=litnet_raw.files, duplicates=litnet_raw.duplicates, log=lambda m: None)
    with pytest.raises(FileExistsError):
        L.prepare(litnet_raw.cfg, files=litnet_raw.files, duplicates=litnet_raw.duplicates, log=lambda m: None)


def test_prepare_rejects_a_changed_raw_file(litnet_raw):
    files = dict(litnet_raw.files)
    size, _ = files["B_FLOWS.zip"]
    files["B_FLOWS.zip"] = (size, "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        L.prepare(litnet_raw.cfg, files=files, duplicates=litnet_raw.duplicates, log=lambda m: None)
    assert not L.processed_dir(litnet_raw.cfg).exists()


def test_prepare_skips_only_a_true_duplicate(litnet_raw):
    with pytest.raises(ValueError, match="different csv"):
        L.prepare(litnet_raw.cfg, files=litnet_raw.files, duplicates={"B_FLOWS.zip": "A_FLOWS.zip"},
                  log=lambda m: None)


def test_prepare_rejects_overlapping_captures(litnet_overlapping):
    with pytest.raises(ValueError, match="overlap"):
        L.prepare(litnet_overlapping.cfg, files=litnet_overlapping.files, duplicates=litnet_overlapping.duplicates,
                  log=lambda m: None)
    assert not L.processed_dir(litnet_overlapping.cfg).exists()


def test_load_refuses_another_time_column(litnet_raw):
    L.prepare(litnet_raw.cfg, files=litnet_raw.files, duplicates=litnet_raw.duplicates, log=lambda m: None)
    cfg = {"dataset": {**litnet_raw.cfg["dataset"], "time_column": "ts"}}
    with pytest.raises(ValueError, match="sorted by te"):
        L.load(cfg)


def test_download_keeps_only_matching_files(litnet_raw, tmp_path):
    source = tmp_path / "published"
    source.mkdir()
    (source / "X_FLOWS.zip").write_bytes(b"published bytes")
    url = f"{source.as_uri()}/{{name}}"
    size = len(b"published bytes")
    good = {"X_FLOWS.zip": (size, hashlib.sha256(b"published bytes").hexdigest())}
    bad = {"X_FLOWS.zip": (size, "0" * 64)}
    with pytest.raises(ValueError, match="sha256"):
        L.download(litnet_raw.cfg, files=bad, url=url, log=lambda m: None)
    assert not any(p.name.startswith("X_FLOWS") for p in litnet_raw.raw.iterdir())
    L.download(litnet_raw.cfg, files=good, url=url, log=lambda m: None)
    assert (litnet_raw.raw / "X_FLOWS.zip").read_bytes() == b"published bytes"
    assert not (litnet_raw.raw / "X_FLOWS.zip.part").exists()
