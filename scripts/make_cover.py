"""Cover image for the submission: one frame with the instruction across the top.

    python scripts/make_cover.py --contact-sheet        # pick a moment
    python scripts/make_cover.py --at 4.5               # render that moment

The default moment is a guess. Run --contact-sheet first, count along, and pass
--at; the sheet is labelled with the timestamp of each tile.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

TITLE = "set the table and pour me some water"
SUB = "two SO-101 arms . VLM planner on OpenVINO . 100/100 randomised seeds"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="out/eval_vlm/video/seed000.mp4")
    ap.add_argument("--at", type=float, default=5.0, help="seconds into the clip")
    ap.add_argument("--out", default="out/cover.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--font", default="C:/Windows/Fonts/segoeui.ttf")
    ap.add_argument("--contact-sheet", action="store_true")
    a = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH", file=sys.stderr)
        return 2
    font = a.font.replace("\\", "/").replace(":", r"\:")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if a.contact_sheet:
        sheet = str(pathlib.Path(a.out).with_name("cover_contact_sheet.png"))
        vf = (f"fps=2,scale=320:-1,"
              f"drawtext=fontfile='{font}':text='%{{pts\\:hms}}':"
              f"x=6:y=6:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6,"
              f"tile=5x4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.clip,
                        "-vf", vf, "-frames:v", "1", sheet], check=True)
        print(f"wrote {sheet} -- pick a moment, then rerun with --at <seconds>")
        return 0

    def esc(s: str) -> str:
        return s.replace(":", r"\:").replace("'", "")

    vf = (f"scale={a.width}:-2,"
          f"drawbox=y=0:w=iw:h=118:color=black@0.62:t=fill,"
          f"drawtext=fontfile='{font}':text='{esc(TITLE)}':"
          f"x=44:y=28:fontsize=42:fontcolor=white,"
          f"drawtext=fontfile='{font}':text='{esc(SUB)}':"
          f"x=46:y=82:fontsize=20:fontcolor=0xBFD4E8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(a.at),
                    "-i", a.clip, "-vf", vf, "-frames:v", "1", a.out], check=True)
    print(f"wrote {a.out} from {a.clip} at t={a.at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
