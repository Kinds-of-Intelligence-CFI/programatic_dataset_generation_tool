from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests._viewer_data import build_image_dataset
from viewer.server import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(build_image_dataset(tmp_path / "image_ds")))


def test_create_app_missing_dataset_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "nope")


def test_get_manifest(client: TestClient):
    resp = client.get("/api/manifest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "image_ds"
    assert body["n_stimuli"] == 2


def test_get_stimuli_returns_all_with_derived_fields(client: TestClient):
    resp = client.get("/api/stimuli")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    first = rows[0]
    assert first["spec_index"] is not None
    assert first["modality"] in {"img", "txt", "audio", "video", "doc"}


def test_get_single_stimulus(client: TestClient):
    resp = client.get("/api/stimuli/0")
    assert resp.status_code == 200
    assert resp.json()["sample_id"] == "0"


def test_get_single_stimulus_unknown_id_404(client: TestClient):
    assert client.get("/api/stimuli/does-not-exist").status_code == 404


def test_get_asset_serves_image_bytes(client: TestClient):
    resp = client.get("/assets/inline/0_0_1.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert len(resp.content) > 0


def test_get_asset_missing_404(client: TestClient):
    assert client.get("/assets/inline/nope.png").status_code == 404


def test_get_asset_rejects_traversal(client: TestClient):
    # Percent-encoded slashes keep the payload inside the /assets/ path segment
    # so it actually reaches the asset handler (a raw ../ is normalized by the
    # client to a different route before it is sent). It must not escape the
    # assets dir and leak manifest.json.
    resp = client.get("/assets/..%2f..%2fmanifest.json")
    assert resp.status_code == 404


def test_root_serves_html(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_unknown_client_route_falls_back_to_spa(client: TestClient):
    # A client-side route (e.g. deep link to a stimulus page) must return the
    # app shell, not a 404, so the frontend router can handle it.
    resp = client.get("/stimuli/5")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
