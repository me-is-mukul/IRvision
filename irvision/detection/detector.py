"""YOLOv8n-OBB detector (pre-trained on DOTA: planes, ships, storage tanks, bridges, ...).

    det = AerialDetector.load("outputs/pretrained/yolov8n-obb.pt", device="cuda")
    boxes = det.detect(rgb)       # rgb (3, H, W) in [0, 1]

DOTA imagery is ~0.1-1 m per pixel; Landsat is 30 m, so most DOTA objects are
smaller than a pixel here. Expect few detections (README D-019).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AerialDetector:
    def __init__(self, model, device: str = "cpu", confidence: float = 0.25, image_size: int = 1024,
                 brightness_gamma: float = 1.8, tile_size: int = 256):
        self.model, self.device = model, device
        self.confidence, self.image_size, self.gamma = confidence, image_size, brightness_gamma
        self.tile_size = tile_size
        self.class_names = model.names

    @classmethod
    def load(cls, model_path: str | Path, device: str = "cpu", **kwargs) -> "AerialDetector":
        from ultralytics import YOLO

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found; run scripts/download_pretrained.py")
        return cls(YOLO(str(model_path)), device, **kwargs)

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Detections as dicts: ``class_name, confidence, box (x1, y1, x2, y2), polygon (4 x 2)``.

        The image is cut into ``tile_size`` tiles (no overlap). YOLO resizes each tile
        to ``image_size`` (4x for 256 -> 1024), so small objects get as many pixels as
        possible; objects cut by a tile border may be missed. Coordinates are in the
        input image's pixels.
        """
        _, h, w = rgb.shape
        size = self.tile_size or max(h, w)
        found = []
        for top in range(0, h, size):
            for left in range(0, w, size):
                for d in self._detect_tile(rgb[:, top : top + size, left : left + size]):
                    x1, y1, x2, y2 = d["box"]
                    d["box"] = [x1 + left, y1 + top, x2 + left, y2 + top]
                    d["polygon"] = [[x + left, y + top] for x, y in d["polygon"]]
                    found.append(d)
        return found

    def _detect_tile(self, rgb: np.ndarray) -> list[dict]:
        """One YOLO call. The tile is brightened with the display gamma first: our RGB is
        scaled to reflectance 0-0.3 and looks dark, unlike the detector's training images."""
        img = np.clip(np.nan_to_num(np.moveaxis(rgb, 0, -1)), 0, 1) ** (1.0 / self.gamma)
        bgr = np.ascontiguousarray((img[..., ::-1] * 255).round().astype(np.uint8))
        result = self.model.predict(bgr, imgsz=self.image_size, conf=self.confidence,
                                    device=self.device, verbose=False)[0]
        if result.obb is None or len(result.obb) == 0:
            return []
        obb = result.obb
        out = []
        for cls, conf, box, poly in zip(obb.cls.tolist(), obb.conf.tolist(), obb.xyxy.cpu().numpy(),
                                        obb.xyxyxyxy.cpu().numpy()):
            out.append({"class_name": self.class_names[int(cls)], "confidence": round(float(conf), 4),
                        "box": [round(float(v), 1) for v in box], "polygon": poly.round(1).tolist()})
        return out
