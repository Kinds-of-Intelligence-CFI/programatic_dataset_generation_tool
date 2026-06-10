import base64
import hashlib
import json
import random
from pathlib import Path

import pytest

from dataset.stimulus import (
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentText,
    ContentVideo,
    Message,
    Stimulus,
)
from dataset.writer import (
    AssetCache,
    normalize_content,
    write_dataset,
    write_jsonl,
    write_manifest,
)
from generation.generate import SampleSpec
from generation.glossary import DemandDefinition, Glossary

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
FAKE_PNG_B64 = base64.b64encode(FAKE_PNG).decode("ascii")
DATA_URI_PNG = f"data:image/png;base64,{FAKE_PNG_B64}"


# ---- normalize_content ------------------------------------------------------


def _normalize(content, tmp_path, cache=None, *, sample_id="s_0", msg=0, idx=0):
    if cache is None:
        cache = AssetCache()
    return normalize_content(
        content,
        sample_id=sample_id,
        message_index=msg,
        content_index=idx,
        output_dir=tmp_path,
        cache=cache,
    )


def test_normalize_text_passes_through(tmp_path: Path):
    c = ContentText(text="hello")
    out = _normalize(c, tmp_path)
    assert out is c


def test_normalize_data_uri_writes_inline_and_rewrites_reference(tmp_path: Path):
    c = ContentImage(image=DATA_URI_PNG)
    out = _normalize(c, tmp_path, sample_id="s_001", msg=0, idx=1)
    assert isinstance(out, ContentImage)
    assert out.image == "assets/inline/s_001_0_1.png"
    on_disk = tmp_path / out.image
    assert on_disk.is_file()
    assert on_disk.read_bytes() == FAKE_PNG


def test_normalize_data_uri_preserves_detail(tmp_path: Path):
    c = ContentImage(image=DATA_URI_PNG, detail="high")
    out = _normalize(c, tmp_path)
    assert isinstance(out, ContentImage)
    assert out.detail == "high"


def test_normalize_data_uri_distinct_indices_get_distinct_files(tmp_path: Path):
    cache = AssetCache()
    out_a = _normalize(
        ContentImage(image=DATA_URI_PNG), tmp_path, cache=cache, msg=0, idx=1
    )
    out_b = _normalize(
        ContentImage(image=DATA_URI_PNG), tmp_path, cache=cache, msg=0, idx=3
    )
    assert out_a.image != out_b.image
    assert (tmp_path / out_a.image).is_file()
    assert (tmp_path / out_b.image).is_file()


def test_normalize_absolute_path_copies_to_files_with_hash_suffix(tmp_path: Path):
    source = tmp_path / "src" / "grid.png"
    source.parent.mkdir()
    source.write_bytes(FAKE_PNG)

    out = _normalize(ContentImage(image=str(source)), tmp_path)
    assert isinstance(out, ContentImage)
    expected_hash = hashlib.sha256(FAKE_PNG).hexdigest()[:12]
    assert out.image == f"assets/files/grid__{expected_hash}.png"
    assert (tmp_path / out.image).read_bytes() == FAKE_PNG


def test_normalize_same_path_twice_dedups(tmp_path: Path):
    source = tmp_path / "src" / "grid.png"
    source.parent.mkdir()
    source.write_bytes(FAKE_PNG)

    cache = AssetCache()
    a = _normalize(ContentImage(image=str(source)), tmp_path, cache=cache)
    b = _normalize(ContentImage(image=str(source)), tmp_path, cache=cache)
    assert a.image == b.image
    files_dir = tmp_path / "assets" / "files"
    assert len(list(files_dir.iterdir())) == 1


def test_normalize_same_bytes_different_basenames_dedups(tmp_path: Path):
    source_a = tmp_path / "src" / "alpha.png"
    source_b = tmp_path / "src" / "beta.png"
    source_a.parent.mkdir()
    source_a.write_bytes(FAKE_PNG)
    source_b.write_bytes(FAKE_PNG)

    cache = AssetCache()
    a = _normalize(ContentImage(image=str(source_a)), tmp_path, cache=cache)
    b = _normalize(ContentImage(image=str(source_b)), tmp_path, cache=cache)
    assert a.image == b.image  # first basename wins
    assert "alpha" in a.image
    files_dir = tmp_path / "assets" / "files"
    assert len(list(files_dir.iterdir())) == 1


def test_normalize_http_url_unchanged(tmp_path: Path):
    c = ContentImage(image="http://example.com/foo.png")
    out = _normalize(c, tmp_path)
    assert out.image == "http://example.com/foo.png"
    assert not (tmp_path / "assets").exists()


def test_normalize_https_url_unchanged(tmp_path: Path):
    c = ContentImage(image="https://example.com/foo.png")
    out = _normalize(c, tmp_path)
    assert out.image == "https://example.com/foo.png"


def test_normalize_already_relative_assets_path_is_idempotent(tmp_path: Path):
    c = ContentImage(image="assets/inline/s_0_0_1.png")
    out = _normalize(c, tmp_path)
    assert out.image == "assets/inline/s_0_0_1.png"


def test_normalize_unsafe_sample_id_raises(tmp_path: Path):
    c = ContentImage(image=DATA_URI_PNG)
    with pytest.raises(ValueError, match="sample_id"):
        _normalize(c, tmp_path, sample_id="bad id/with/slash")


def test_normalize_missing_source_file_raises(tmp_path: Path):
    c = ContentImage(image=str(tmp_path / "does_not_exist.png"))
    with pytest.raises(FileNotFoundError, match="does_not_exist.png"):
        _normalize(c, tmp_path)


# ---- normalize_content for audio / video / document ------------------------

FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmtfake"
FAKE_WAV_B64 = base64.b64encode(FAKE_WAV).decode("ascii")
DATA_URI_WAV = f"data:audio/wav;base64,{FAKE_WAV_B64}"

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42fake"
FAKE_MP4_B64 = base64.b64encode(FAKE_MP4).decode("ascii")
DATA_URI_MP4 = f"data:video/mp4;base64,{FAKE_MP4_B64}"

FAKE_PDF = b"%PDF-1.4 fake"
FAKE_PDF_B64 = base64.b64encode(FAKE_PDF).decode("ascii")
DATA_URI_PDF = f"data:application/pdf;base64,{FAKE_PDF_B64}"


def test_normalize_audio_data_uri_writes_inline_wav_and_preserves_format(tmp_path: Path):
    c = ContentAudio(audio=DATA_URI_WAV, format="wav")
    out = _normalize(c, tmp_path, sample_id="s_001", msg=0, idx=2)
    assert isinstance(out, ContentAudio)
    assert out.audio == "assets/inline/s_001_0_2.wav"
    assert (tmp_path / out.audio).read_bytes() == FAKE_WAV
    assert out.format == "wav"


def test_normalize_video_data_uri_writes_inline_mp4_and_preserves_format(tmp_path: Path):
    c = ContentVideo(video=DATA_URI_MP4, format="mp4")
    out = _normalize(c, tmp_path, sample_id="s_001", msg=0, idx=1)
    assert isinstance(out, ContentVideo)
    assert out.video == "assets/inline/s_001_0_1.mp4"
    assert (tmp_path / out.video).read_bytes() == FAKE_MP4
    assert out.format == "mp4"


def test_normalize_document_data_uri_writes_inline_pdf_and_preserves_fields(tmp_path: Path):
    c = ContentDocument(
        document=DATA_URI_PDF, filename="report.pdf", mime_type="application/pdf"
    )
    out = _normalize(c, tmp_path, sample_id="s_001", msg=0, idx=0)
    assert isinstance(out, ContentDocument)
    assert out.document == "assets/inline/s_001_0_0.pdf"
    assert (tmp_path / out.document).read_bytes() == FAKE_PDF
    assert out.filename == "report.pdf"
    assert out.mime_type == "application/pdf"


def test_normalize_audio_path_copies_to_files_with_hash_suffix(tmp_path: Path):
    src = tmp_path / "src" / "clip.wav"
    src.parent.mkdir()
    src.write_bytes(FAKE_WAV)

    out = _normalize(ContentAudio(audio=str(src), format="wav"), tmp_path)
    assert isinstance(out, ContentAudio)
    h = hashlib.sha256(FAKE_WAV).hexdigest()[:12]
    assert out.audio == f"assets/files/clip__{h}.wav"
    assert (tmp_path / out.audio).read_bytes() == FAKE_WAV
    assert out.format == "wav"


def test_normalize_video_path_copies_to_files_with_hash_suffix(tmp_path: Path):
    src = tmp_path / "src" / "scene.mp4"
    src.parent.mkdir()
    src.write_bytes(FAKE_MP4)

    out = _normalize(ContentVideo(video=str(src), format="mp4"), tmp_path)
    assert isinstance(out, ContentVideo)
    h = hashlib.sha256(FAKE_MP4).hexdigest()[:12]
    assert out.video == f"assets/files/scene__{h}.mp4"
    assert out.format == "mp4"


def test_normalize_document_path_copies_to_files_preserving_optional_fields(tmp_path: Path):
    src = tmp_path / "src" / "doc.pdf"
    src.parent.mkdir()
    src.write_bytes(FAKE_PDF)

    out = _normalize(
        ContentDocument(
            document=str(src), filename="doc.pdf", mime_type="application/pdf"
        ),
        tmp_path,
    )
    assert isinstance(out, ContentDocument)
    h = hashlib.sha256(FAKE_PDF).hexdigest()[:12]
    assert out.document == f"assets/files/doc__{h}.pdf"
    assert out.filename == "doc.pdf"
    assert out.mime_type == "application/pdf"


def test_normalize_audio_https_url_unchanged(tmp_path: Path):
    c = ContentAudio(audio="https://example.com/clip.wav", format="wav")
    out = _normalize(c, tmp_path)
    assert out.audio == "https://example.com/clip.wav"
    assert out.format == "wav"
    assert not (tmp_path / "assets").exists()


def test_normalize_video_already_relative_assets_path_is_idempotent(tmp_path: Path):
    c = ContentVideo(video="assets/files/clip__abcdefabcdef.mp4", format="mp4")
    out = _normalize(c, tmp_path)
    assert out.video == "assets/files/clip__abcdefabcdef.mp4"
    assert out.format == "mp4"


def test_normalize_document_already_relative_assets_path_is_idempotent(tmp_path: Path):
    c = ContentDocument(document="assets/inline/s_0_0_0.pdf")
    out = _normalize(c, tmp_path)
    assert isinstance(out, ContentDocument)
    assert out.document == "assets/inline/s_0_0_0.pdf"


def test_normalize_dedups_across_modalities_when_bytes_match(tmp_path: Path):
    # Same bytes appearing under two different modalities should produce one
    # file in assets/files/. The cache is hash-keyed, not type-keyed.
    audio_src = tmp_path / "src" / "a.wav"
    video_src = tmp_path / "src" / "v.mp4"
    audio_src.parent.mkdir()
    audio_src.write_bytes(FAKE_WAV)
    video_src.write_bytes(FAKE_WAV)

    cache = AssetCache()
    a = _normalize(ContentAudio(audio=str(audio_src), format="wav"), tmp_path, cache=cache)
    v = _normalize(ContentVideo(video=str(video_src), format="mp4"), tmp_path, cache=cache)
    assert a.audio == v.video
    files_dir = tmp_path / "assets" / "files"
    assert len(list(files_dir.iterdir())) == 1


# ---- write_jsonl / write_dataset --------------------------------------------


def _stimulus(sample_id: str, target: str = "x") -> Stimulus:
    return Stimulus(
        sample_id=sample_id,
        spec=SampleSpec(demands={"cap_a": 1}, params={"n": 1}),
        messages=[Message(role="user", content="hello")],
        target=target,
    )


def test_write_jsonl_streams_one_line_per_stimulus(tmp_path: Path):
    path = tmp_path / "stimuli.jsonl"
    stimuli = [_stimulus(f"s_{i}") for i in range(4)]
    n = write_jsonl(path, iter(stimuli), output_dir=tmp_path, cache=AssetCache())
    assert n == 4
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    for raw, expected in zip(lines, stimuli):
        d = json.loads(raw)
        assert d["sample_id"] == expected.sample_id


def test_write_jsonl_consumes_iterable_lazily(tmp_path: Path):
    consumed = []

    def gen():
        for i in range(3):
            consumed.append(i)
            yield _stimulus(f"s_{i}")

    write_jsonl(tmp_path / "out.jsonl", gen(), output_dir=tmp_path, cache=AssetCache())
    assert consumed == [0, 1, 2]


def test_write_manifest_contains_all_required_fields(tmp_path: Path):
    path = tmp_path / "manifest.json"
    specs = [SampleSpec(demands={"b": 2, "a": 1}, params={"k": 1})]
    write_manifest(
        path,
        name="my_dataset",
        specs=specs,
        global_seed=42,
        n_reps=3,
        n_stimuli=3,
    )
    m = json.loads(path.read_text())
    assert m["name"] == "my_dataset"
    assert m["global_seed"] == 42
    assert m["n_reps"] == 3
    assert m["n_stimuli"] == 3
    assert "library_version" in m
    assert "timestamp" in m
    assert m["specs"] == [{"demands": {"b": 2, "a": 1}, "params": {"k": 1}}]


def test_write_manifest_includes_functional_when_supplied(tmp_path: Path):
    path = tmp_path / "manifest.json"
    specs = [SampleSpec(demands={"cap_a": 1}, params={"k": 1})]
    functional = SampleSpec(demands={"shared_cap": 2}, params={"shared_key": 7})
    write_manifest(
        path, name="x", specs=specs, global_seed=0, n_reps=1, n_stimuli=1,
        functional=functional,
    )
    m = json.loads(path.read_text())
    assert m["functional"] == {"demands": {"shared_cap": 2}, "params": {"shared_key": 7}}
    assert m["specs"] == [{"demands": {"cap_a": 1}, "params": {"k": 1}}]


def test_write_manifest_functional_is_null_when_not_supplied(tmp_path: Path):
    path = tmp_path / "manifest.json"
    write_manifest(
        path, name="x", specs=[SampleSpec()], global_seed=0, n_reps=1, n_stimuli=1,
    )
    m = json.loads(path.read_text())
    assert m["functional"] is None


def test_write_manifest_includes_glossary_when_supplied(tmp_path: Path):
    path = tmp_path / "manifest.json"
    glossary = Glossary(
        name="g",
        tree={
            "Language": {
                "creation": DemandDefinition(
                    description="produce", paper_link="http://example.com/p"
                )
            }
        },
    )
    write_manifest(
        path, name="x", specs=[SampleSpec()], global_seed=0, n_reps=1, n_stimuli=1,
        glossary=glossary,
    )
    m = json.loads(path.read_text())
    assert m["glossary"]["name"] == "g"
    demand = m["glossary"]["demands"]["creation"]
    assert demand["description"] == "produce"
    assert demand["paper_link"] == "http://example.com/p"
    assert demand["ancestry"] == ["Language"]


def test_write_manifest_glossary_is_null_when_not_supplied(tmp_path: Path):
    path = tmp_path / "manifest.json"
    write_manifest(
        path, name="x", specs=[SampleSpec()], global_seed=0, n_reps=1, n_stimuli=1,
    )
    m = json.loads(path.read_text())
    assert m["glossary"] is None


def test_write_dataset_end_to_end_with_inline_image(tmp_path: Path):
    rng = random.Random(0xC0FFEE)
    output_dir = tmp_path / "demo_dataset"
    output_dir.mkdir()

    specs = [SampleSpec(demands={"cap_a": 1}, params={"i": i}) for i in range(5)]

    def build_stimuli():
        for i, spec in enumerate(specs):
            messages: list[Message] = [
                Message(role="user", content=f"q-{rng.randint(0, 999)}"),
            ]
            if i == 0:
                messages.append(
                    Message(
                        role="assistant",
                        content=[ContentImage.from_bytes(FAKE_PNG, suffix="png")],
                    )
                )
            yield Stimulus(
                sample_id=f"s_{i:04d}",
                spec=spec,
                messages=messages,
                target=rng.choice(["a", "b", "c"]),
                metadata={"rep_index": 0},
            )

    write_dataset(
        output_dir,
        name="demo_dataset",
        stimuli=build_stimuli(),
        specs=specs,
        global_seed=0xC0FFEE,
        n_reps=1,
    )

    jsonl = output_dir / "stimuli.jsonl"
    manifest = output_dir / "manifest.json"
    inline_dir = output_dir / "assets" / "inline"

    assert jsonl.is_file()
    assert manifest.is_file()
    assert inline_dir.is_dir()

    lines = jsonl.read_text().splitlines()
    assert len(lines) == 5

    first = json.loads(lines[0])
    image_content = first["messages"][1]["content"][0]
    assert image_content["type"] == "image"
    assert image_content["image"].startswith("assets/inline/")
    assert (output_dir / image_content["image"]).read_bytes() == FAKE_PNG

    m = json.loads(manifest.read_text())
    assert m["n_stimuli"] == 5
    assert m["n_reps"] == 1
    assert m["global_seed"] == 0xC0FFEE
    assert len(m["specs"]) == 5
