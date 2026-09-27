"""Web API: endpoints return the payload the frontend expects (light pipeline, no model files)."""

import copy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from irvision.api import server
from irvision.api.payload import STAGES, build_payload, file_writer
from irvision.inference.pipeline import IRVisionPipeline
from irvision.models.baseline import GrayColorizer
from irvision.utils.config import load_config
from tests.conftest import textured_field


@pytest.fixture
def client(monkeypatch):
    cfg = copy.deepcopy(load_config())
    for stage in ("segmentation", "super_resolution", "detection"):
        cfg["optional"][stage]["enabled"] = False
    pipeline = IRVisionPipeline(cfg, colorizer=GrayColorizer(), device="cpu")
    monkeypatch.setattr(server, "get_pipeline", lambda: pipeline)
    return TestClient(server.app)


def png_bytes(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["colorizer"] == "gray"


def test_examples_list_demo_folder(client):
    examples = client.get("/api/examples").json()
    assert {e["id"] for e in examples} >= {"hyderabad_farmland", "hyderabad_lakes", "hyderabad_city"}
    assert all(e["thumbnail"].startswith("data:image/webp;base64,") for e in examples)


def test_process_example_payload(client):
    body = client.post("/api/process/example/hyderabad_farmland").json()
    assert [s["key"] for s in body["stages"]] == [k for k, _, _ in STAGES]
    status = {s["key"]: s["status"] for s in body["stages"]}
    assert status["colorization"] == "done" and status["super_resolution"] == "off"
    assert {"input", "enhanced", "colorized", "reference"} <= set(body["images"])
    assert 0 < body["metrics"]["ssim"] <= 1 and body["metrics"]["psnr"] > 0
    assert body["shape"] == [512, 512]


def test_unknown_example_404(client):
    assert client.post("/api/process/example/nope").status_code == 404


def test_upload_with_and_without_reference(client):
    ir = (textured_field(96) * 60000).astype(np.uint16)
    files = {"image": ("ir.png", png_bytes(ir), "image/png")}
    body = client.post("/api/process", files=files).json()
    assert "psnr" not in body["metrics"] and body["images"]["colorized"].startswith("data:image/webp")

    ref = (np.random.default_rng(0).random((96, 96, 3)) * 255).astype(np.uint8)
    files["reference"] = ("ref.png", png_bytes(ref), "image/png")
    body = client.post("/api/process", files=files).json()
    assert "psnr" in body["metrics"] and "reference" in body["images"]


def test_upload_rejects_bad_files(client):
    bad = client.post("/api/process", files={"image": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 422
    tiny = client.post("/api/process", files={"image": ("t.png", png_bytes(np.zeros((8, 8), np.uint8)), "image/png")})
    assert tiny.status_code == 422


def test_file_writer_payload(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["optional"]["segmentation"]["enabled"] = False
    result = IRVisionPipeline(cfg, colorizer=GrayColorizer(), device="cpu").process_image(
        (290 + 30 * textured_field(64)).astype(np.float32))
    payload = build_payload(result, file_writer(tmp_path, "/demo/x"))
    assert payload["images"]["colorized"] == "/demo/x/colorized.webp"
    assert (tmp_path / "colorized.webp").exists()
