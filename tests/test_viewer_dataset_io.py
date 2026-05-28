from pathlib import Path

import pytest

from tests._viewer_data import build_image_dataset, build_text_dataset
from viewer import dataset_io


@pytest.fixture
def image_dataset(tmp_path: Path) -> Path:
    return build_image_dataset(tmp_path / "image_ds")


@pytest.fixture
def text_dataset(tmp_path: Path) -> Path:
    return build_text_dataset(tmp_path / "text_ds")


# ---- resolve / load --------------------------------------------------------


def test_resolve_dataset_exposes_expected_paths(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    assert paths.manifest == image_dataset / "manifest.json"
    assert paths.stimuli == image_dataset / "stimuli.jsonl"
    assert paths.assets == image_dataset / "assets"


def test_resolve_dataset_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        dataset_io.resolve_dataset(tmp_path / "does_not_exist")


def test_resolve_dataset_missing_manifest_raises(tmp_path: Path):
    (tmp_path / "stimuli.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        dataset_io.resolve_dataset(tmp_path)


def test_load_manifest_real_fields(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    manifest = dataset_io.load_manifest(paths)
    assert manifest["name"] == "image_ds"
    assert manifest["n_stimuli"] == 2
    assert manifest["global_seed"] == 7
    assert isinstance(manifest["specs"], list) and manifest["specs"]


# ---- spec_index ------------------------------------------------------------


def test_spec_index_matches_manifest_entry(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    specs = dataset_io.load_manifest(paths)["specs"]
    for record in dataset_io.iter_stimuli(paths):
        idx = record["spec_index"]
        assert idx is not None
        assert dataset_io.canonical_spec(specs[idx]) == dataset_io.canonical_spec(
            record["spec"]
        )


def test_spec_index_distinguishes_specs_differing_by_one_param(text_dataset: Path):
    paths = dataset_io.resolve_dataset(text_dataset)
    records = list(dataset_io.iter_stimuli(paths))
    indices = {r["sample_id"]: r["spec_index"] for r in records}
    assert indices["0"] != indices["1"]


def test_spec_index_handles_empty_demands(text_dataset: Path):
    paths = dataset_io.resolve_dataset(text_dataset)
    manifest = dataset_io.load_manifest(paths)
    control = next(s for s in manifest["specs"] if s["demands"] == [])
    assert dataset_io.canonical_spec(control) == dataset_io.canonical_spec(control)


def test_spec_index_none_when_no_match():
    orphan = {"demands": ["nope"], "params": {"x": 1}}
    assert dataset_io.spec_index_of(orphan, [{"demands": [], "params": {}}]) is None


# ---- modality --------------------------------------------------------------


def test_modality_image_for_image_plus_text():
    record = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "q"},
                    {"type": "image", "image": "assets/inline/0_0_1.png"},
                ],
            }
        ]
    }
    assert dataset_io.modality_of(record) == "img"


def test_modality_image_from_real_dataset(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    for record in dataset_io.iter_stimuli(paths):
        assert record["modality"] == "img"


def test_modality_text_for_text_only(text_dataset: Path):
    paths = dataset_io.resolve_dataset(text_dataset)
    for record in dataset_io.iter_stimuli(paths):
        assert record["modality"] == "txt"


def test_modality_text_for_plain_string_content():
    record = {"messages": [{"role": "user", "content": "just a string"}]}
    assert dataset_io.modality_of(record) == "txt"


# ---- streaming -------------------------------------------------------------


def test_iter_stimuli_yields_every_record(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    records = list(dataset_io.iter_stimuli(paths))
    assert len(records) == 2
    assert all("sample_id" in r for r in records)


def test_iter_stimuli_ignores_blank_lines(tmp_path: Path):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "manifest.json").write_text('{"name": "x", "specs": []}', encoding="utf-8")
    (ds / "stimuli.jsonl").write_text(
        '{"sample_id": "0", "spec": {"demands": [], "params": {}}, '
        '"messages": [], "target": "t"}\n\n  \n',
        encoding="utf-8",
    )
    records = list(dataset_io.iter_stimuli(dataset_io.resolve_dataset(ds)))
    assert len(records) == 1


# ---- safe asset path -------------------------------------------------------


def test_safe_asset_path_resolves_real_asset(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    resolved = dataset_io.safe_asset_path(paths, "inline/0_0_1.png")
    assert resolved is not None and resolved.is_file()


def test_safe_asset_path_rejects_traversal(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    assert dataset_io.safe_asset_path(paths, "../manifest.json") is None
    assert dataset_io.safe_asset_path(paths, "../../etc/passwd") is None


def test_safe_asset_path_none_for_missing_file(image_dataset: Path):
    paths = dataset_io.resolve_dataset(image_dataset)
    assert dataset_io.safe_asset_path(paths, "inline/nope.png") is None
