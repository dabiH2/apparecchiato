"""Cut the ten seed clips into the montage the video script asks for.

Each clip is trimmed to `--per-clip` seconds, captioned with the seed and the
randomisation that seed actually drew, and concatenated. The captions are read
from the scene, not typed in, so they cannot disagree with what is on screen.

    python scripts/build_montage.py --out out/montage.mp4

Needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from apparecchiato.sim.layout import sample_scene


def caption(seed: int) -> str:
    s = sample_scene(seed)
    p, m = s.by_name("plate"), s.by_name("mug")
    return (f"seed {seed:03d}   plate {p.scale:.2f}x {p.mass * 1000:.0f} g "
            f"mu {p.friction:.2f}   mug {m.scale:.2f}x   light {s.light_intensity:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default="out/eval_vlm/video")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--per-clip", type=float, default=3.0)
    ap.add_argument("--out", default="out/montage.mp4")
    ap.add_argument("--width", type=int, default=960)
    # ffmpeg's drawtext needs an explicit font on Windows: the build ships
    # without a fontconfig default and dies with a bare "Cannot load default
    # config file" and an access violation, which reads like a codec problem.
    ap.add_argument("--font", default="C:/Windows/Fonts/consola.ttf")
    a = ap.parse_args()
    # Slashes first, THEN escape the drive colon -- the other order turns
    # "C\:/Windows" into "C/:/Windows" and ffmpeg reports only "Invalid argument".
    font = a.font.replace("\\", "/").replace(":", r"\:")

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH", file=sys.stderr)
        return 2

    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    clips = pathlib.Path(a.clips)
    work = pathlib.Path(a.out).parent / "montage_parts"
    work.mkdir(parents=True, exist_ok=True)

    parts = []
    for seed in seeds:
        src = clips / f"seed{seed:03d}.mp4"
        if not src.exists():
            print(f"missing {src}, skipping", file=sys.stderr)
            continue
        dst = work / f"part{seed:03d}.mp4"
        text = caption(seed).replace(":", r"\:").replace("'", "")
        # Take the LAST `per-clip` seconds: the end of an episode is the part
        # worth showing -- the table laid and the pour finished.
        vf = (f"scale={a.width}:-2,"
              f"drawbox=y=ih-46:w=iw:h=46:color=black@0.55:t=fill,"
              f"drawtext=fontfile='{font}':text='{text}':"
              f"x=18:y=h-32:fontsize=18:fontcolor=white")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-sseof", f"-{a.per_clip}",
             "-i", str(src), "-vf", vf, "-an", "-r", "30", str(dst)],
            check=True)
        parts.append(dst)

    if not parts:
        print("no clips found", file=sys.stderr)
        return 1

    listing = work / "parts.txt"
    listing.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out],
        check=True)
    print(f"wrote {a.out} from {len(parts)} clips "
          f"({len(parts) * a.per_clip:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
