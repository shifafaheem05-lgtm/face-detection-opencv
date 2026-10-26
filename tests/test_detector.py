import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from face_detector import Face, FaceDetector
from face_detector.cli import main


def blank(h=240, w=320):
    return np.full((h, w, 3), 127, dtype=np.uint8)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector()

    def test_blank_image_has_no_faces(self):
        self.assertEqual(self.detector.detect(blank()), [])

    def test_grayscale_input_is_accepted(self):
        gray = cv2.cvtColor(blank(), cv2.COLOR_BGR2GRAY)
        self.assertEqual(self.detector.detect(gray), [])

    def test_empty_frame_raises(self):
        with self.assertRaises(ValueError):
            self.detector.detect(np.empty((0, 0, 3), dtype=np.uint8))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            FaceDetector(scale_factor=1.0)
        with self.assertRaises(ValueError):
            FaceDetector(min_neighbors=0)

    def test_draw_returns_copy_and_marks_box(self):
        frame = blank()
        out = FaceDetector.draw(frame, [Face(50, 50, 80, 80)])
        self.assertTrue((frame == 127).all(), "original must not be modified")
        self.assertFalse((out == frame).all(), "box should be drawn on the copy")

    def test_blur_changes_only_face_region(self):
        frame = blank()
        frame[50:130, 50:130] = np.random.RandomState(0).randint(0, 255, (80, 80, 3), dtype=np.uint8)
        out = FaceDetector.blur(frame, [Face(50, 50, 80, 80)])
        self.assertFalse((out[50:130, 50:130] == frame[50:130, 50:130]).all())
        self.assertTrue((out[:40] == frame[:40]).all())

    def test_even_blur_strength_is_handled(self):
        FaceDetector.blur(blank(), [Face(10, 10, 40, 40)], strength=50)


class CliTests(unittest.TestCase):
    def test_image_command_saves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "in.jpg", Path(tmp) / "out.jpg"
            cv2.imwrite(str(src), blank())
            self.assertEqual(main(["image", str(src), "-o", str(dest)]), 0)
            self.assertTrue(dest.exists())

    def test_missing_image_returns_error(self):
        self.assertEqual(main(["image", "does_not_exist.jpg"]), 1)

    def test_missing_video_returns_error(self):
        self.assertEqual(main(["video", "does_not_exist.mp4"]), 1)

    def test_video_command_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
            w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
            for _ in range(5):
                w.write(blank())
            w.release()
            self.assertEqual(main(["video", str(src), "-o", str(dest)]), 0)
            self.assertTrue(dest.exists())


if __name__ == "__main__":
    unittest.main()
