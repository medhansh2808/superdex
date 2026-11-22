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
    printed_header = False
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
        if "quarter_turn" in result:
            if not printed_header:
                print(" iter  command   physical    start   reached   longest hold    success    drops")
                printed_header = True
            for name, task in result["quarter_turn"].items():
                print(f"{i:5d}  {name:8s} {task['physical_yaw_deg']:+8.2f}°"
                      f" {task['start_stable_rate']:8.1%} {task['target_face_rate']:9.1%}"
                      f" {task['mean_longest_target_hold_s']:12.2f} s"
                      f" {task['success_rate']:10.1%} {task['drop_rate']:8.1%}")
            continue
        if not printed_header:
            print(" iter    phys+°   phys-°   rad/s  yaw drop   tilt success   tilt error   tilt drop   held s (yaw/tilt)")
            printed_header = True
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
