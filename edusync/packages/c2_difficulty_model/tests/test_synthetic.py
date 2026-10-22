"""The per-model synthetic datasets: separate, deterministic, labelled, extractable."""

import random
from pathlib import Path

import pytest

from edusync_c2.difficulty.registry import FEATURE_NAMES
from edusync_c2.difficulty.synthetic import ConceptSpec
from edusync_c2_model import synthetic as syn
from edusync_c2_model.extract import ExtractError
from edusync_c2_model.labels import read_labels
from edusync_c2_model.manifest import read_manifest
from edusync_c2_model.split import make_split

ROWS = 60
SEED = 11


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    return out, syn.build_expert_datasets(out, rows_per_expert=ROWS, base_seed=SEED)


def test_five_separate_datasets_with_exact_row_counts(datasets):
    out, built = datasets
    assert list(built) == ["A", "B", "C", "D", "M"]
    assert {p.name for p in built.values()} == {
        "language",
        "prerequisites",
        "explanation",
        "density",
        "combiner",
    }
    for paths in built.values():
        assert paths.root.parent == out
        assert paths.rows == ROWS
        assert paths.reused is False
        labels = read_labels([paths.labels])
        assert len(labels) == ROWS
        assert all(label.synthetic for label in labels.labels.values())
        assert all(label.n_raters == 2 for label in labels.labels.values())
        assert labels.ties_excluded == 0  # the simulated adjudicator settles disagreements


def test_ids_courses_and_files(datasets):
    _, built = datasets
    for letter, paths in built.items():
        entries = read_manifest(paths.manifest)
        assert len(entries) == paths.decks
        assert all(e.lecture_id.startswith(f"S{letter}-") for e in entries)
        assert len({e.course_id for e in entries}) >= 4
        assert all(e.file.suffix == ".pptx" for e in entries)
        assert (paths.root / "dataset.json").is_file()


def test_graph_usage_differs_per_expert(datasets):
    _, built = datasets
    graphs = {
        letter: [e.graph is not None for e in read_manifest(p.manifest)]
        for letter, p in built.items()
    }
    assert not any(graphs["A"])
    assert not any(graphs["C"])
    assert not any(graphs["D"])
    assert all(graphs["B"])
    assert any(graphs["M"])
    assert not all(graphs["M"])


def test_datasets_differ_across_experts(datasets):
    _, built = datasets
    label_files = {letter: p.labels.read_text(encoding="utf-8") for letter, p in built.items()}
    assert len(set(label_files.values())) == 5
    seeds = {
        letter: {pl.spec.title for pl in syn.plan_dataset(syn.EXPERTS[letter], ROWS, SEED)}
        for letter in built
    }
    for a in seeds:
        for b in seeds:
            if a < b:
                assert not seeds[a] & seeds[b]  # disjoint seed ranges
    shapes = {
        letter: {pl.shape for pl in syn.plan_dataset(syn.EXPERTS[letter], 200, SEED)}
        for letter in built
    }
    assert shapes["A"] == {"language"}
    assert shapes["B"] == {"prerequisites"}
    assert shapes["C"] == {"explanation"}
    assert shapes["D"] == {"density"}
    assert shapes["M"] == {"language", "prerequisites", "explanation", "density"}
    layouts = {
        letter: {pl.layout for pl in syn.plan_dataset(syn.EXPERTS[letter], 200, SEED)}
        for letter in built
    }
    assert layouts["D"] == {True}
    assert layouts["A"] == {False}


def test_shapes_change_the_concept_properties():
    def mean(letter, attr):
        concepts = [
            c for pl in syn.plan_dataset(syn.EXPERTS[letter], 400, SEED) for c in pl.spec.concepts
        ]
        return sum(getattr(c, attr) for c in concepts) / len(concepts)

    assert mean("A", "words") > mean("A", "steps")
    assert mean("C", "steps") > mean("A", "steps")
    assert mean("C", "words") > mean("B", "words")

    def prereqs(letter):
        concepts = [
            c for pl in syn.plan_dataset(syn.EXPERTS[letter], 400, SEED) for c in pl.spec.concepts
        ]
        return sum(len(c.prerequisites) for c in concepts) / len(concepts)

    assert prereqs("B") > prereqs("A")


def test_prerequisite_chains_point_to_the_previous_concept():
    for plan in syn.plan_dataset(syn.EXPERTS["B"], 40, SEED):
        labels = [c.label for c in plan.spec.concepts]
        assert plan.spec.concepts[0].prerequisites == []
        for i, concept in enumerate(plan.spec.concepts[1:], start=1):
            assert labels[i - 1] in concept.prerequisites
            assert set(concept.prerequisites) <= set(labels[:i])


def test_generation_is_deterministic_for_a_seed(tmp_path, datasets):
    _, built = datasets
    again = syn.build_expert_datasets(tmp_path / "again", rows_per_expert=ROWS, base_seed=SEED)
    for letter, paths in built.items():
        other = again[letter]
        assert paths.labels.read_text() == other.labels.read_text()
        assert paths.manifest.read_text() == other.manifest.read_text()
        for a, b in zip(read_manifest(paths.manifest), read_manifest(other.manifest), strict=True):
            assert (a.graph is None) == (b.graph is None)
            if a.graph:
                assert a.graph.read_bytes() == b.graph.read_bytes()
    different = syn.build_expert_datasets(
        tmp_path / "different", rows_per_expert=ROWS, base_seed=SEED + 1, experts=("A",)
    )
    assert different["A"].labels.read_text() != built["A"].labels.read_text()


def test_existing_dataset_is_reused_or_needs_force(tmp_path):
    first = syn.build_expert_datasets(tmp_path, rows_per_expert=16, base_seed=1, experts=("A",))
    deck = read_manifest(first["A"].manifest)[0].file
    stamp = deck.stat().st_mtime_ns
    second = syn.build_expert_datasets(tmp_path, rows_per_expert=16, base_seed=1, experts=("A",))
    assert second["A"].reused is True
    assert deck.stat().st_mtime_ns == stamp
    with pytest.raises(syn.SyntheticError, match="different parameters"):
        syn.build_expert_datasets(tmp_path, rows_per_expert=24, base_seed=1, experts=("A",))
    replaced = syn.build_expert_datasets(
        tmp_path, rows_per_expert=24, base_seed=1, experts=("A",), force=True
    )
    assert replaced["A"].rows == 24
    assert replaced["A"].reused is False
    (replaced["A"].root / "dataset.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(syn.SyntheticError, match="different parameters"):
        syn.build_expert_datasets(tmp_path, rows_per_expert=24, base_seed=1, experts=("A",))


def test_a_dataset_missing_its_labels_is_not_reused(tmp_path):
    built = syn.build_expert_datasets(tmp_path, rows_per_expert=16, base_seed=1, experts=("C",))
    built["C"].labels.unlink()
    with pytest.raises(syn.SyntheticError, match="different parameters"):
        syn.build_expert_datasets(tmp_path, rows_per_expert=16, base_seed=1, experts=("C",))


def test_argument_errors(tmp_path):
    with pytest.raises(syn.SyntheticError, match="unknown expert"):
        syn.build_expert_datasets(tmp_path, 8, experts=("Z",))
    with pytest.raises(syn.SyntheticError, match="jobs"):
        syn.build_expert_datasets(tmp_path, 8, jobs=0)
    with pytest.raises(syn.SyntheticError, match="rows"):
        syn.plan_dataset(syn.EXPERTS["A"], 0, 1)


def test_parallel_build_matches_serial(tmp_path):
    serial = syn.build_expert_datasets(tmp_path / "s", 24, 5, experts=("D",))
    parallel = syn.build_expert_datasets(tmp_path / "p", 24, 5, jobs=2, experts=("D",))
    assert serial["D"].labels.read_text() == parallel["D"].labels.read_text()
    assert serial["D"].decks == parallel["D"].decks


def test_label_csv_follows_section_9_and_passes_validate_labels(datasets, c2_script):
    _, built = datasets
    assert syn.LABEL_COLUMNS == c2_script("export_label_sheet").COLUMNS
    validate = c2_script("validate_labels")
    for paths in built.values():
        header, rows = validate.read_rows(paths.labels)
        findings, labelled = validate.check_rows(header, rows)
        assert findings.count("error") == 0
        agreement = validate.agreement(labelled)
        pooled = agreement["pooled"]
        assert pooled["pairs"] == ROWS
        assert 0.5 < pooled["exact_agreement"] < 1.0  # raters mostly agree, sometimes not
        assert pooled["qwk"] is not None


def test_label_rows_with_certain_noise_and_none():
    spec = syn.plan_dataset(syn.EXPERTS["A"], 8, 3)[0]
    quiet = syn.DeckPlan(**{**spec.__dict__, "label_noise": 0.0, "rater_flip": 0.0})
    rows = syn.label_rows(quiet)
    assert len(rows) == 2 * len(spec.spec.concepts)  # R1 + R2, no adjudication
    assert {r[6] for r in rows} == {"R1", "R2"}
    noisy = syn.DeckPlan(**{**spec.__dict__, "label_noise": 1.0, "rater_flip": 1.0})
    rows = syn.label_rows(noisy)
    assert len(rows) == 3 * len(spec.spec.concepts)  # every concept is adjudicated
    assert all(r[12] == "true" for r in rows)
    adjudicated = [r for r in rows if r[6] == "ADJ"]
    first = [r for r in rows if r[6] == "R1"]
    assert [r[5] for r in adjudicated] == [r[5] for r in first]  # ADJ follows rater 1


def test_adjacent_flip_moves_exactly_one_class():
    rng = random.Random(0)
    assert syn._adjacent(0, rng) == 1
    assert syn._adjacent(2, rng) == 1
    assert {syn._adjacent(1, rng) for _ in range(40)} == {0, 2}


def test_graph_decks_use_slug_ids_and_others_casefolded_labels():
    graph_plan = syn.plan_dataset(syn.EXPERTS["B"], 8, 3)[0]
    plain_plan = syn.plan_dataset(syn.EXPERTS["A"], 8, 3)[0]
    first_graph = syn.label_rows(graph_plan)[0]
    first_plain = syn.label_rows(plain_plan)[0]
    assert first_graph[1] == first_graph[2]  # concept_id == slug
    assert first_plain[1] == first_plain[3].casefold()


def test_extracted_features_join_every_label_and_differ_by_expert(datasets, tmp_path):
    _, built = datasets
    semantic = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("expl.semantic")]
    prereq = FEATURE_NAMES.index("prereq.direct_count")
    assert semantic
    results = {}
    for letter, paths in built.items():
        examples, joined, stats = syn.load_dataset(paths, tmp_path / "cache" / paths.name)
        assert len(examples) == ROWS
        assert joined.unmatched_labels == []
        assert stats.misses == paths.decks
        assert stats.hits == 0
        assert all(e.synthetic and e.n_raters == 2 for e in examples)
        # the semantic scorer is off, so extraction never needs it
        assert all(e.statuses[i] == "unavailable" for e in examples for i in semantic)
        results[letter] = (examples, joined)
    assert all(e.statuses[prereq] == "computed" for e in results["B"][0])
    assert all(e.statuses[prereq] != "computed" for e in results["A"][0])
    again = syn.load_dataset(built["B"], tmp_path / "cache" / "prerequisites")
    assert again[2].hits == built["B"].decks
    assert again[2].misses == 0
    assert again[0] == results["B"][0]


def test_class_balance():
    assert syn.class_balance([]) == {"Low": 0.0, "Medium": 0.0, "High": 0.0}


def test_class_balance_of_each_dataset_has_all_three_classes(datasets, tmp_path):
    _, built = datasets
    for paths in built.values():
        examples = syn.load_dataset(paths, tmp_path / paths.name)[0]
        share = syn.class_balance(examples)
        assert sum(share.values()) == pytest.approx(1.0)
        assert min(share.values()) > 0.0


def test_a_group_split_of_a_dataset_has_no_leakage(datasets):
    _, built = datasets
    entries = read_manifest(built["M"].manifest)
    split = make_split({e.lecture_id: e.course_id for e in entries}, seed=1)
    assert split["unit"] == "course"


# ---------------------------------------------------------------------------------------- CLI


def test_cli_generates_extracts_and_reuses(tmp_path, capsys, package_script):
    cli = package_script("make_expert_datasets")
    args = ["--out", str(tmp_path / "out"), "--rows-per-expert", "24", "--extract"]
    args += ["--cache", str(tmp_path / "cache")]
    assert cli.main(args) == 0
    first = capsys.readouterr().out
    assert "SYNTHETIC DATA" in first
    assert "(written)" in first
    assert "0h/" in first
    for name in ("language", "prerequisites", "explanation", "density", "combiner"):
        assert name in first
    assert cli.main(args) == 0
    second = capsys.readouterr().out
    assert "(reused)" in second
    assert "/0m" in second  # every lecture came from the cache


def test_cli_without_extract(tmp_path, capsys, package_script):
    cli = package_script("make_expert_datasets")
    assert cli.main(["--out", str(tmp_path), "--rows-per-expert", "8"]) == 0
    assert "unmatched" not in capsys.readouterr().out


def test_cli_errors(tmp_path, capsys, package_script):
    cli = package_script("make_expert_datasets")
    assert cli.main(["--out", str(tmp_path), "--rows-per-expert", "0"]) == 2
    assert "rows" in capsys.readouterr().err
    assert cli.main(["--bad-flag"]) == 2
    assert cli.main(["--help"]) == 0
    capsys.readouterr()
    out = str(tmp_path / "x")
    assert cli.main(["--out", out, "--rows-per-expert", "8"]) == 0
    assert cli.main(["--out", out, "--rows-per-expert", "16"]) == 2
    assert "different parameters" in capsys.readouterr().err
    assert cli.main(["--out", out, "--rows-per-expert", "16", "--force"]) == 0


def test_cli_reports_extraction_errors(tmp_path, capsys, monkeypatch, package_script):
    cli = package_script("make_expert_datasets")

    def boom(*_args, **_kwargs):
        raise ExtractError("boom")

    monkeypatch.setattr(cli, "load_dataset", boom)
    assert cli.main(["--out", str(tmp_path), "--rows-per-expert", "8", "--extract"]) == 2
    assert "boom" in capsys.readouterr().err


def test_default_directories_are_inside_the_package(package_script):
    package_root = Path(syn.__file__).resolve().parents[2]
    assert package_root / "data" / "synthetic" == syn.DEFAULT_DATA_DIR
    cli = package_script("make_expert_datasets")
    assert package_root / "cache" / "synthetic" == cli.DEFAULT_CACHE_DIR


def test_concept_specs_stay_valid_after_shaping():
    for letter in syn.EXPERTS:
        for plan in syn.plan_dataset(syn.EXPERTS[letter], 80, SEED):
            for concept in plan.spec.concepts:
                assert isinstance(concept, ConceptSpec)
                assert concept.words >= 0
                assert 0 <= concept.bullet_depth <= 3
