"""Show simulator evaluations from one run, including the policy before training."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing train.log")
    args = parser.parse_args()
    print(f"Run {args.run.name}")
    print(" iter    phys+°   phys-°   rad/s  yaw drop   tilt success   tilt error   tilt drop   held s (yaw/tilt)")
    last_iteration = 0
    for line in (args.run / "train.log").read_text().splitlines():
        label, _, payload = line.partition(" ")
        if label == "iteration":
            match = re.match(r"(\d+)/\d+ ", payload)
            if match:
                last_iteration = int(match[1])
        if label not in ("untrained_policy", "intermediate_policy", "trained_policy"):
            continue
        result = json.loads(payload)
        i = result.get("iteration", last_iteration if label == "trained_policy" else 0)
        yaw, tilt = result["yaw"], result["reorient"]
        by = yaw["by_command"]
        held = (f"{yaw['mean_held_s']:.2f}/{tilt['mean_held_s']:.2f}"
                if "mean_held_s" in yaw and "mean_held_s" in tilt else "—")
        if "physical_yaw_deg" in by["positive"]:
            for name in ("positive", "negative"):
                cmd = by[name]
                print(f"  turn {i:3d} {name:8s} physical {cmd['physical_yaw_deg']:+7.2f}°"
                      f"  first face {cmd['first_face_rate']:.1%}  complete {cmd['full_turn_success']:.1%}")
        positive = by["positive"]["signed_rotation_deg"]
        negative = -by["negative"]["signed_rotation_deg"]
        print(f"{i:5d} {positive:8.2f} "
              f"{negative:8.2f} {yaw['signed_rad_s_while_held']:7.3f} "
              f"{yaw['drop_rate']:9.1%} {tilt['success_rate']:14.1%} "
              f"{tilt['median_final_held_error_deg']:11.2f}° {tilt['drop_rate']:10.1%}   {held}")


if __name__ == "__main__":
    main()
