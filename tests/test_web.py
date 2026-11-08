"""Smoke tests for the Streamlit demo (skipped if Streamlit is not installed)."""
import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

APP = str(Path(__file__).resolve().parent.parent / "web" / "app.py")


@unittest.skipIf(AppTest is None, "streamlit not installed")
class WebAppTests(unittest.TestCase):
    def test_sample_image_detects_a_face(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value, "1")

    def test_blur_mode_runs(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.toggle[0].set_value(True).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value, "1")


if __name__ == "__main__":
    unittest.main()
