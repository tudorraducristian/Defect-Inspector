"""Check the station before photos: the zone, a cross at its centre, the light and a score.

  station_check.py [--profile profiles/piulita-m10] [--auto-exposure]

The zone and the camera come from the profile, so a light that has no photos yet can be checked.
A window shows the camera with the zone of interest, the band where the light is measured and a
cross at the centre. Every few seconds it writes on the image, and prints, the brightness of the
zone's border, how much its sides differ and the score of the piece. --auto-exposure leaves the
exposure to the camera instead of the profile's value. Esc closes it. Laptop only.
"""
import json
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from defect_inspector.backends import load_backend
from defect_inspector.camera import CameraWorker, settings_for_this_machine
from defect_inspector.inspector import Inspector
from defect_inspector.profile import load_profile

MARGIN = 40
WINDOW = "statia"
SIDES = ("sus", "jos", "stanga", "dreapta")


def border_brightness(gray, roi, margin=MARGIN):
    """Mean of the zone's border band, and of each of its four sides."""
    x, y, width, height = roi
    zone = gray[y:y + height, x:x + width].astype(np.float32)
    band = np.zeros(zone.shape, bool)
    band[:margin, :] = band[-margin:, :] = True
    band[:, :margin] = band[:, -margin:] = True
    parts = (zone[:margin, :], zone[-margin:, :], zone[:, :margin], zone[:, -margin:])
    return float(zone[band].mean()), {name: float(part.mean()) for name, part in zip(SIDES, parts)}


def check_settings(fields, auto_exposure):
    """Camera settings and zone from a profile.json; with auto_exposure the camera sets its own exposure."""
    settings = settings_for_this_machine(fields.get("camera"))
    if auto_exposure:
        settings = settings._replace(exposure=None)
    return settings, fields.get("roi")


def _measure(frame, roi, inspector):
    mean, sides = border_brightness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), roi)
    text = "fundal {:.1f}  laturi {:.1f}".format(mean, max(sides.values()) - min(sides.values()))
    verdict = inspector.inspect(frame)
    text += "  scor {:.3f} / prag {:.3f}  {}".format(verdict.score, verdict.threshold, verdict.label)
    return text


def _draw(frame, roi, text):
    x, y, width, height = roi
    cx, cy = x + width // 2, y + height // 2
    shown = frame.copy()
    cv2.rectangle(shown, (x, y), (x + width, y + height), (0, 0, 255), 3)
    cv2.rectangle(shown, (x + MARGIN, y + MARGIN), (x + width - MARGIN, y + height - MARGIN), (0, 200, 255), 1)
    cv2.line(shown, (cx - 60, cy), (cx + 60, cy), (0, 255, 0), 2)
    cv2.line(shown, (cx, cy - 60), (cx, cy + 60), (0, 255, 0), 2)
    cv2.putText(shown, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
    cv2.putText(shown, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    return shown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, default=Path("profiles/piulita-m10"))
    parser.add_argument("--auto-exposure", action="store_true")
    args = parser.parse_args(argv)
    fields = json.loads((args.profile / "profile.json").read_text(encoding="utf-8"))
    settings, roi = check_settings(fields, args.auto_exposure)
    inspector = Inspector(load_profile(args.profile), load_backend("ort-cpu", str(args.profile / "model.onnx")))
    state = {"frame": None, "text": "masor...", "running": True}
    lock = threading.Lock()

    def measure_forever():
        # The model takes seconds on the laptop's processor, so it runs beside the window.
        while state["running"]:
            with lock:
                frame = state["frame"]
            if frame is None:
                time.sleep(0.2)
                continue
            text = _measure(frame, roi, inspector)
            print(time.strftime("%H:%M:%S"), text, flush=True)
            with lock:
                state["text"] = text
            time.sleep(1.0)

    camera = CameraWorker(settings)
    camera.start()
    threading.Thread(target=measure_forever, daemon=True).start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    try:
        while True:
            latest = camera.latest()
            if latest is not None:
                with lock:
                    state["frame"] = latest[0].copy()
                    text = state["text"]
                cv2.imshow(WINDOW, _draw(latest[0], roi, text))
            if cv2.waitKey(30) == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 0
    finally:
        state["running"] = False
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
