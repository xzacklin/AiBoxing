"""
Boxer AI - Phase 3: Labeled clip recorder.

Same pose-overlaid webcam loop as Phase 2, plus a tiny recorder. Press a
number key to start a recording with a class label, throw the move, press
space to stop. The clip is saved as a single .npz file under data/raw/.

Controls (with the video window focused):
    1     start recording a JAB
    2     start recording a CROSS
    3     start recording a HOOK
    4     start recording an UPPERCUT
    5     start recording IDLE  (stance, breathing, non-punch movement)
    space stop and save the current recording
    q     quit (an in-progress recording is discarded)

Each saved .npz contains:
    xy:    (T, 17, 2) float32   pixel coords of the 17 COCO keypoints
    conf:  (T, 17)    float32   per-keypoint confidence in [0, 1]
    times: (T,)       float64   seconds since recording start
    label: str                  class label
    fps:   float                mean recording FPS for the clip
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KP_CONF_THRESHOLD = 0.5

# Maps the keyboard key code to the class label it starts a recording for.
LABEL_KEYS = {
    ord("1"): "jab",
    ord("2"): "cross",
    ord("3"): "hook",
    ord("4"): "uppercut",
    ord("5"): "idle",
}

# Resolve data/raw/ relative to this file, not the user's cwd, so it works
# no matter where they run the script from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def draw_skeleton(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray) -> None:
    for i, j in SKELETON_EDGES:
        if conf[i] < KP_CONF_THRESHOLD or conf[j] < KP_CONF_THRESHOLD:
            continue
        p1 = (int(xy[i, 0]), int(xy[i, 1]))
        p2 = (int(xy[j, 0]), int(xy[j, 1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
    for idx in range(len(xy)):
        if conf[idx] < KP_CONF_THRESHOLD:
            continue
        x, y = int(xy[idx, 0]), int(xy[idx, 1])
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)


def pick_primary_person(xys: np.ndarray, confs: np.ndarray):
    """If more than one person is detected, take the one with the highest
    mean keypoint confidence. Returns (xy_17x2, conf_17) or (None, None)."""
    if len(xys) == 0:
        return None, None
    if len(xys) == 1:
        return xys[0], confs[0]
    mean_confs = confs.mean(axis=1)
    idx = int(np.argmax(mean_confs))
    return xys[idx], confs[idx]


def save_clip(frames_xy, frames_conf, frames_t, label: str) -> None:
    if len(frames_xy) == 0:
        print("Empty clip, nothing to save.")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    xy = np.stack(frames_xy).astype(np.float32)
    conf = np.stack(frames_conf).astype(np.float32)
    times = np.array(frames_t, dtype=np.float64)
    duration = float(times[-1] - times[0]) if times[-1] > times[0] else 0.0
    fps = len(times) / duration if duration > 0 else 0.0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = DATA_DIR / f"{label}_{stamp}.npz"
    np.savez_compressed(
        path,
        xy=xy,
        conf=conf,
        times=times,
        label=label,
        fps=fps,
    )
    print(f"Saved {label:8s} -> {path.name}   ({len(xy)} frames, {fps:.1f} FPS)")


def draw_hud(frame: np.ndarray, recording: bool, current_label, n_frames: int) -> None:
    h, w = frame.shape[:2]
    if recording:
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 200), -1)
        text = (
            f"REC  [{str(current_label).upper()}]   "
            f"frames: {n_frames}   (space = stop)"
        )
        cv2.putText(
            frame, text, (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    else:
        cv2.rectangle(frame, (0, h - 40), (w, h), (40, 40, 40), -1)
        cv2.putText(
            frame,
            "1:jab  2:cross  3:hook  4:uppercut  5:idle    q:quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )


def main() -> None:
    print("Loading pose model...")
    model = YOLO("yolov8n-pose.pt")
    print("Loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print(f"Clips will be saved under: {DATA_DIR}")
    print("Press 1-5 to start, space to stop, q to quit.")

    prev_time = time.time()
    fps = 0.0

    recording = False
    current_label = None
    rec_xy: list = []
    rec_conf: list = []
    rec_t: list = []
    rec_start = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
        frame = cv2.flip(frame, 1)

        results = model(frame, verbose=False)
        result = results[0]

        person_xy, person_conf = None, None
        if (
            result.keypoints is not None
            and result.keypoints.conf is not None
            and len(result.keypoints) > 0
        ):
            xys = result.keypoints.xy.cpu().numpy()
            confs = result.keypoints.conf.cpu().numpy()
            person_xy, person_conf = pick_primary_person(xys, confs)

        if person_xy is not None:
            draw_skeleton(frame, person_xy, person_conf)

        # If recording, append this frame's pose data. When no person is
        # detected we still append zeros + conf=0 so the time axis stays
        # evenly spaced -- that matters for the temporal model later.
        if recording:
            t = time.time() - rec_start
            if person_xy is not None:
                rec_xy.append(person_xy.copy())
                rec_conf.append(person_conf.copy())
            else:
                rec_xy.append(np.zeros((17, 2), dtype=np.float32))
                rec_conf.append(np.zeros((17,), dtype=np.float32))
            rec_t.append(t)

        # FPS overlay
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        draw_hud(frame, recording, current_label, len(rec_xy))

        cv2.imshow("Boxer AI - Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:  # no key pressed
            continue
        if key == ord("q"):
            if recording:
                print("Discarding in-progress recording.")
            break
        if recording and key == ord(" "):
            save_clip(rec_xy, rec_conf, rec_t, current_label)
            recording = False
            current_label = None
            rec_xy, rec_conf, rec_t = [], [], []
        elif not recording and key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            recording = True
            rec_start = time.time()
            rec_xy, rec_conf, rec_t = [], [], []
            print(f"Recording started: {current_label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()