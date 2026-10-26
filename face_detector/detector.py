"""Core face detection logic (Haar cascade, OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

CASCADE_FILE = "haarcascade_frontalface_default.xml"


@dataclass(frozen=True)
class Face:
    """A detected face as a bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


class FaceDetector:
    """Detect frontal faces in BGR images or video frames.

    Parameters
    ----------
    scale_factor:  how much the image is shrunk at each pyramid step (>1).
                   Smaller = more accurate but slower.
    min_neighbors: how many overlapping detections are needed to keep a face.
                   Higher = fewer false positives.
    min_size:      smallest face (width, height) to look for, in pixels.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 1:
            raise ValueError("min_neighbors must be at least 1")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_FILE)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load OpenCV cascade file: {CASCADE_FILE}")

    def detect(self, frame: np.ndarray) -> List[Face]:
        """Return all faces found in `frame` (BGR or grayscale)."""
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.equalizeHist(gray)  # improves detection in uneven lighting

        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [Face(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

    @staticmethod
    def draw(
        frame: np.ndarray,
        faces: List[Face],
        color: Tuple[int, int, int] = (0, 200, 0),
        label: bool = True,
    ) -> np.ndarray:
        """Return a copy of `frame` with a box (and number) around each face."""
        out = frame.copy()
        for i, f in enumerate(faces, start=1):
            cv2.rectangle(out, (f.x, f.y), (f.x + f.w, f.y + f.h), color, 2)
            if label:
                cv2.putText(
                    out, f"Face {i}", (f.x, max(f.y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )
        return out

    @staticmethod
    def blur(frame: np.ndarray, faces: List[Face], strength: int = 51) -> np.ndarray:
        """Return a copy of `frame` with every face blurred (privacy mode)."""
        k = strength if strength % 2 == 1 else strength + 1  # kernel must be odd
        out = frame.copy()
        for f in faces:
            roi = out[f.y : f.y + f.h, f.x : f.x + f.w]
            out[f.y : f.y + f.h, f.x : f.x + f.w] = cv2.GaussianBlur(roi, (k, k), 0)
        return out
