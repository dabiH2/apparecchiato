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
# NOTE: do not put "VLM planner" back in this line. The VLM's plan is rejected on every
# seed and the README says so; the 100/100 is the deterministic fallback's. This string is
# the one sentence a judge is guaranteed to read, so it must survive its own README.
SUB = "two SO-101 arms . plans validated against physics . 100/100 seeds"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="out/eval_clips/video/seed000.mp4")
    ap.add_argument("--at", type=float, default=5.0, help="seconds into the clip")
    ap.add_argument("--out", default="out/cover.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--font", default="C:/Windows/Fonts/segoeui.ttf")
    ap.add_argument("--contact-sheet", action="store_true")
    ap.add_argument("--aspect", default="16:9", choices=("16:9", "source"),
                    help="16:9 is what the lablab submission form asks for")
    ap.add_argument("--crop-y", type=int, default=64,
                    help="top edge of the 16:9 crop within the source frame")
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

    # Crop to 16:9 BEFORE scaling. lablab's submission form requires 16:9 and the
    # simulator renders 4:3, so a straight scale would have been rejected or
    # letterboxed by whoever opened it. The crop is biased downward (`--crop-y`)
    # because the interesting half of a 4:3 frame here is the table, not the
    # ceiling: a centred crop takes an equal bite out of both and loses the near
    # edge of the place setting.
    crop = ""
    if a.aspect == "16:9":
        crop = f"crop=iw:iw*9/16:0:{a.crop_y},"

    vf = (f"{crop}scale={a.width}:-2,"
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
