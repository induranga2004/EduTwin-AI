"""The committed demo script, run on tiny synthetic datasets with the hash-based fake encoder."""

from edusync_c2_model.report import SYNTHETIC_BANNER


def test_demo_trains_every_model_and_writes_a_bannered_report(package_script, toy, tmp_path):
    script = package_script("train_synthetic_demo")
    console, report = script.run(
        tmp_path / "data",
        tmp_path / "cache",
        toy.hash_embed,
        "fake",
        rows=48,
        seed=3,
        n_boot=5,
    )
    assert report.splitlines()[2].startswith(f"**{SYNTHETIC_BANNER}")
    for name in ("combiner", "language", "prerequisites", "explanation", "density"):
        assert f"## {name} dataset: test split" in report
    for row in ("rule baseline", "majority class", "feature logistic", "group ensemble"):
        assert row in report
    assert "expert A alone" in report
    assert "human-human" in report
    assert "runtime" in console


def test_demo_main_reports_missing_model_weights(package_script, tmp_path, capsys):
    script = package_script("train_synthetic_demo")
    code = script.main(["--model-dir", str(tmp_path / "nowhere")])
    assert code == 2
    assert "cannot be loaded" in capsys.readouterr().err
    assert script.main(["--bad-flag"]) == 2


def test_demo_main_writes_the_report(package_script, toy, tmp_path, monkeypatch, capsys):
    script = package_script("train_synthetic_demo")
    monkeypatch.setattr(script, "encoder_embedder", lambda model_dir: (toy.hash_embed, "fake"))
    out = tmp_path / "runs"
    code = script.main(
        [
            "--data", str(tmp_path / "data"),
            "--cache", str(tmp_path / "cache"),
            "--out", str(out),
            "--rows-per-expert", "48",
            "--seed", "3",
            "--bootstrap", "5",
        ]
    )  # fmt: skip
    assert code == 0
    assert (out / "train_synthetic_demo.md").read_text(encoding="utf-8").startswith("# C2")
    assert "SYNTHETIC DATA" in capsys.readouterr().out
    # same folder, different parameters: the dataset error becomes exit code 2
    again = ["--data", str(tmp_path / "data"), "--cache", str(tmp_path / "cache")]
    assert script.main([*again, "--out", str(out), "--rows-per-expert", "40"]) == 2
    assert "different parameters" in capsys.readouterr().err
