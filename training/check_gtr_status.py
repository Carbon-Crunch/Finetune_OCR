#!/usr/bin/env python3
"""
=============================================================================
  GTR Rotation Status Checker (`check_gtr_status.py`)
=============================================================================
This script quickly scans all `gtr_*.json` files in `all_gtr/` and reports:
  - Total GTR files found
  - Done (already have `rotation_needed` and `degree` labels added)
  - Remaining (need orientation labels added)
  - Degree breakdown (0°, 90°, 180°, 270°)
  - Single-page vs Multi-page document counts

Usage:
------
  ./.venv/bin/python3 check_gtr_status.py
  ./.venv/bin/python3 check_gtr_status.py --gtr-dir ./all_gtr
=============================================================================
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import Counter


def check_status(gtr_dir: Path) -> None:
    if not gtr_dir.exists():
        print(f"\033[91m[ERROR] Directory not found: {gtr_dir}\033[0m")
        sys.exit(1)

    gtr_files = sorted(gtr_dir.glob("gtr_*.json"))
    total = len(gtr_files)

    if total == 0:
        print(f"\033[93m[WARNING] No gtr_*.json files found in {gtr_dir}\033[0m")
        return

    done_count = 0
    remaining_count = 0
    corrupted_count = 0

    degree_counts = Counter()
    multi_page_count = 0
    single_page_count = 0
    remaining_files = []
    corrupted_files = []

    for path in gtr_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            is_done = False
            deg = None

            if isinstance(data, dict):
                if "rotation_needed" in data and "degree" in data:
                    is_done = True
                    deg = data.get("degree", 0)
                    if "page_orientations" in data and len(data["page_orientations"]) > 1:
                        multi_page_count += 1
                    else:
                        single_page_count += 1
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                if "rotation_needed" in data[0] and "degree" in data[0]:
                    is_done = True
                    deg = data[0].get("degree", 0)
                    if "page_orientations" in data[0] and len(data[0]["page_orientations"]) > 1:
                        multi_page_count += 1
                    else:
                        single_page_count += 1

            if is_done:
                done_count += 1
                if deg is not None:
                    degree_counts[deg] += 1
            else:
                remaining_count += 1
                if len(remaining_files) < 15:
                    remaining_files.append(path.name)

        except Exception as exc:
            corrupted_count += 1
            if len(corrupted_files) < 20:
                corrupted_files.append(f"{path.name} -> {exc}")
            if len(remaining_files) < 15:
                remaining_files.append(f"{path.name} (Error: {exc})")

    # Progress bar
    percent = (done_count / total) * 100 if total > 0 else 0
    bar_len = 30
    filled_len = int(bar_len * done_count // total) if total > 0 else 0
    bar = "█" * filled_len + "-" * (bar_len - filled_len)

    print("\n" + "=" * 65)
    print("                GTR ROTATION LOGIC STATUS REPORT")
    print("=" * 65)
    print(f" Directory Checked : {gtr_dir}")
    print(f" Total GTR Files   : \033[1m{total}\033[0m")
    print("-----------------------------------------------------------------")
    print(f" Progress          : [{bar}] \033[1;32m{percent:.1f}%\033[0m")
    print(f" ✅ DONE           : \033[1;32m{done_count}\033[0m / {total}")
    print(f" ⏳ REMAINING      : \033[1;33m{remaining_count}\033[0m / {total}")
    if corrupted_count > 0:
        print(f" ❌ CORRUPTED/ERR  : \033[1;31m{corrupted_count}\033[0m / {total}")
    print("-----------------------------------------------------------------")

    if done_count > 0:
        print(" 📊 DONE BREAKDOWN BY ROTATION DEGREE:")
        for deg in sorted(degree_counts.keys()):
            cnt = degree_counts[deg]
            pct = (cnt / done_count) * 100
            deg_str = f"{deg}° CW"
            print(f"    • {deg_str:<8}: {cnt:>4} files ({pct:.1f}%)")
        print()
        print(f" 📄 DOCUMENT TYPE BREAKDOWN:")
        print(f"    • Single-Page Documents : {single_page_count}")
        print(f"    • Multi-Page Documents  : {multi_page_count}")
        print("-----------------------------------------------------------------")

    if corrupted_count > 0:
        print(f"\n ❌ CORRUPTED/ERROR FILES ({corrupted_count} total):")
        for err_info in corrupted_files:
            print(f"    - {err_info}")
        print("-----------------------------------------------------------------")

    if remaining_count > 0:
        print(f" ⏳ NEXT REMAINING FILES TO PROCESS (Showing up to 15):")
        for name in remaining_files:
            print(f"    - {name}")
        if remaining_count > len(remaining_files):
            print(f"    ... and {remaining_count - len(remaining_files)} more files.")
        print("\n 👉 To update remaining files, run:")
        print("    ./.venv/bin/python3 update_gtr_orientation.py --workers 6")
    else:
        print("\n 🎉 ALL GTR FILES ARE 100% DONE! No rotation updates remaining.")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Check progress of GTR orientation updates.")
    parser.add_argument("--gtr-dir", type=str, default="./all_gtr", help="Path to directory containing gtr_*.json files")
    args = parser.parse_args()

    check_status(Path(args.gtr_dir).resolve())


if __name__ == "__main__":
    main()
