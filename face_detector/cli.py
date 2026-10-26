"""Command-line interface: image, video and webcam face detection."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .detector import FaceDetector


def _render(detector: FaceDetector, frame, blur: bool):
    faces = detector.detect(frame)
    out = detector.blur(frame, faces) if blur else detector.draw(frame, faces)
    return faces, out


def run_image(args) -> int:
    src = Path(args.input)
    frame = cv2.imread(str(src))
    if frame is None:
        print(f"Error: cannot read image '{src}'. Check the path and file type.", file=sys.stderr)
        return 1

    detector = FaceDetector(args.scale, args.neighbors, (args.min_size, args.min_size))
    faces, out = _render(detector, frame, args.blur)

    dest = Path(args.output) if args.output else Path("output") / f"{src.stem}_detected{src.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), out)

    print(f"Faces found: {len(faces)}")
    print(f"Saved result to: {dest}")
    if args.show:
        cv2.imshow("Face Detection - press any key", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_video(args) -> int:
    src = Path(args.input)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Error: cannot open video '{src}'.", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dest = Path(args.output) if args.output else Path("output") / f"{src.stem}_detected.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = FaceDetector(args.scale, args.neighbors, (args.min_size, args.min_size))
    frames = with_faces = most = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces, out = _render(detector, frame, args.blur)
        writer.write(out)
        frames += 1
        with_faces += bool(faces)
        most = max(most, len(faces))
        if total and frames % 50 == 0:
            print(f"  processed {frames}/{total} frames", end="\r")

    cap.release()
    writer.release()
    print(f"\nFrames processed: {frames}")
    print(f"Frames with at least one face: {with_faces}")
    print(f"Most faces in a single frame: {most}")
    print(f"Saved result to: {dest}")
    return 0


def run_webcam(args) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}.", file=sys.stderr)
        return 1

    detector = FaceDetector(args.scale, args.neighbors, (args.min_size, args.min_size))
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    blur = args.blur
    prev = time.time()
    print("Webcam started. Keys: [q] quit  [s] save snapshot  [b] toggle blur")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: lost camera feed.", file=sys.stderr)
            break

        faces, out = _render(detector, frame, blur)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(out, f"Faces: {len(faces)}  FPS: {fps:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Face Detection (q to quit)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            blur = not blur
        if key == ord("s"):
            path = out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), out)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scale", type=float, default=1.1, help="scale factor (default 1.1)")
    common.add_argument("--neighbors", type=int, default=5, help="min neighbors (default 5)")
    common.add_argument("--min-size", type=int, default=30, help="smallest face in pixels (default 30)")
    common.add_argument("--blur", action="store_true", help="blur faces instead of drawing boxes")

    parser = argparse.ArgumentParser(
        prog="face_detector",
        description="Detect faces in images, videos or a live webcam using OpenCV.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("image", parents=[common], help="detect faces in an image")
    p_img.add_argument("input", help="path to the image")
    p_img.add_argument("-o", "--output", help="where to save the result")
    p_img.add_argument("--show", action="store_true", help="open a window with the result")
    p_img.set_defaults(func=run_image)

    p_vid = sub.add_parser("video", parents=[common], help="detect faces in a video file")
    p_vid.add_argument("input", help="path to the video")
    p_vid.add_argument("-o", "--output", help="where to save the result")
    p_vid.set_defaults(func=run_video)

    p_cam = sub.add_parser("webcam", parents=[common], help="live detection from the webcam")
    p_cam.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p_cam.set_defaults(func=run_webcam)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
