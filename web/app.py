"""Browser demo for the face detector: upload a photo or use the camera."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st

from face_detector.detector import FaceDetector

MAX_SIDE = 1280
SAMPLE = ROOT / "samples" / "astronaut.jpg"

st.set_page_config(page_title="Face Detection Demo", page_icon="👁️", layout="wide")


def decode(data: bytes) -> np.ndarray | None:
    """Decode uploaded bytes into a BGR image, shrinking very large photos."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


st.title("👁️ Face Detection Demo")
st.caption("OpenCV Haar cascade, running in your browser session. "
           "Images are processed in memory and are never stored.")

with st.sidebar:
    st.header("Detection settings")
    scale = st.slider("Scale factor", 1.02, 1.5, 1.1, 0.01,
                      help="Smaller = more accurate but slower.")
    neighbors = st.slider("Min neighbours", 1, 10, 5,
                          help="Higher = fewer false positives.")
    min_size = st.slider("Min face size (px)", 20, 200, 30, 5)
    blur = st.toggle("Privacy blur mode", help="Blur faces instead of drawing boxes.")

source = st.radio("Image source", ["Sample image", "Upload a photo", "Use camera"],
                  horizontal=True)

image = None
if source == "Sample image":
    if SAMPLE.exists():
        image = decode(SAMPLE.read_bytes())
        st.caption("Sample: NASA astronaut portrait (public domain).")
    else:
        st.warning("Sample image not found.")
elif source == "Upload a photo":
    up = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"])
    if up is not None:
        image = decode(up.getvalue())
        if image is None:
            st.error("That file could not be read as an image.")
else:
    shot = st.camera_input("Take a photo")
    if shot is not None:
        image = decode(shot.getvalue())

if image is not None:
    detector = FaceDetector(scale_factor=scale, min_neighbors=neighbors,
                            min_size=(min_size, min_size))
    faces = detector.detect(image)
    result = detector.blur(image, faces) if blur else detector.draw(image, faces)

    c1, c2 = st.columns(2)
    c1.subheader("Original")
    c1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    c2.subheader("Blurred" if blur else "Detected")
    c2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), width="stretch")

    st.metric("Faces found", len(faces))
    if not faces:
        st.info("No faces found. Try a front-facing, well-lit photo or lower "
                "'Min neighbours' / 'Min face size'.")
    ok, png = cv2.imencode(".png", result)
    if ok:
        st.download_button("Download result", png.tobytes(), "face_result.png", "image/png")

st.divider()
st.markdown("Source code: [github.com/shifafaheem05-lgtm/face-detection-opencv]"
            "(https://github.com/shifafaheem05-lgtm/face-detection-opencv)")
