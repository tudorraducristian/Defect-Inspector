"""Score the test images of a dataset with a profile and report the results for each light.

  evaluate_lights.py --profile profiles/piulita-m10 --dataset data/piulita-m10-lumini-ready

The images in test/ are already cut to the zone, so they are scored like Inspector does, without
cutting again. A name starting with its light (L3_piece-07_shot-01.png) is counted under that light;
other names under "-". Marker and scratch are the visible defects of the spec; other types are
reported on their own. Laptop only.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

from defect_inspector.backends import load_backend
from defect_inspector.postprocessing import score_from_map
from defect_inspector.preprocessing import to_model_input
from defect_inspector.profile import load_profile

VISIBLE = ("marker", "zgarietura")
ALL = "toate"


def light_of(name):
    prefix = name.split("_", 1)[0]
    return prefix if prefix.startswith("L") and "_" in name else "-"


def auroc(good, defect):
    """How often a defective image scores above a good one; a tie counts as half."""
    if not good or not defect:
        return None
    wins = sum((d > g) + 0.5 * (d == g) for d in defect for g in good)
    return wins / float(len(good) * len(defect))


def _summary(rows, threshold, visible):
    good = [score for _, kind, score in rows if kind == "good"]
    defect = [score for _, kind, score in rows if kind != "good"]
    detected = {}
    for _, kind, score in rows:
        if kind != "good":
            found = detected.setdefault(kind, [0, 0])
            found[0] += int(score > threshold)
            found[1] += 1
    shown = [detected[kind] for kind in visible if kind in detected]
    total_visible = sum(n for _, n in shown)
    false_alarms = sum(score > threshold for score in good)
    return {
        "good": len(good),
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarms / float(len(good)) if good else None,
        "detected": detected,
        "visible_rate": sum(k for k, _ in shown) / float(total_visible) if total_visible else None,
        "auroc": auroc(good, defect),
    }


def summarize(rows, threshold, visible=VISIBLE):
    """rows: (light, kind, score). A score strictly above the threshold means defect, as at runtime."""
    lights = sorted(set(light for light, _, _ in rows))
    result = {light: _summary([r for r in rows if r[0] == light], threshold, visible) for light in lights}
    result[ALL] = _summary(rows, threshold, visible)
    return result


def score_dataset(profile_folder, dataset):
    profile = load_profile(Path(profile_folder))
    backend = load_backend("ort-cpu", str(Path(profile_folder) / "model.onnx"))
    rows = []
    test_root = Path(dataset) / "test"
    for kind_folder in sorted(p for p in test_root.iterdir() if p.is_dir()):
        for path in sorted(kind_folder.glob("*.png")):
            image = cv2.imread(str(path))
            x = to_model_input(image, profile.image_size, profile.mean, profile.std)
            rows.append((light_of(path.name), kind_folder.name, float(score_from_map(backend.run(x)))))
    return rows, profile.threshold


def _print_table(summary, threshold):
    print("prag {:.4f}".format(threshold))
    print("{:<6} {:>14} {:>16} {:>7}  pe tipuri".format("lumina", "alarme false", "defecte vizibile", "AUROC"))
    for light, s in summary.items():
        alarms = "{}/{}".format(s["false_alarms"], s["good"])
        visible = "-" if s["visible_rate"] is None else "{:.0%}".format(s["visible_rate"])
        score = "-" if s["auroc"] is None else "{:.3f}".format(s["auroc"])
        kinds = ", ".join("{} {}/{}".format(k, v[0], v[1]) for k, v in sorted(s["detected"].items()))
        print("{:<6} {:>14} {:>16} {:>7}  {}".format(light, alarms, visible, score, kinds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--threshold", type=float, help="instead of the profile's, to see what another gives")
    parser.add_argument("--json", type=Path, help="also write the scores and the summary here")
    args = parser.parse_args(argv)
    rows, threshold = score_dataset(args.profile, args.dataset)
    if args.threshold is not None:
        threshold = args.threshold
    summary = summarize(rows, threshold)
    _print_table(summary, threshold)
    if args.json:
        args.json.write_text(json.dumps({"threshold": threshold, "summary": summary, "rows": rows}, indent=2),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
