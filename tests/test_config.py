from forge.config import load_config


def test_inherit_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 0\na:\n  b: 1\n  c: 2\n")
    (tmp_path / "exp.yaml").write_text("inherit: [base.yaml]\na:\n  b: 5\n")
    cfg = load_config(tmp_path / "exp.yaml", ["a.c=7", "seed=3"])
    assert cfg == {"seed": 3, "a": {"b": 5, "c": 7}}
