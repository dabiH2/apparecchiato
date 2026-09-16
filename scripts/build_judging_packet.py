"""Assemble `judging/` -- everything a review session needs, in one folder.

Run this after ANY change to the results, the README or the video. A packet that
has drifted from the repo is worse than no packet: the last review scored the
submission down for three claims that were true of the code and false of the
README, and a stale copy is how that happens twice.

    python scripts/build_judging_packet.py
    python scripts/build_judging_packet.py --check    # fail if anything is stale

INDEX.md is hand-written and is left alone; everything else is copied.
"""
from __future__ import annotations

import argparse
import filecmp
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / "hackathon-docs"


# source -> path inside judging/. A missing source is reported, not fatal: the
# packet should still build on a machine that has not rendered the video.
ITEMS: list[tuple[pathlib.Path, str]] = [
    (ROOT / "README.md", "README.md"),
    (ROOT / "docs/06-openvino-findings.md", "docs/06-openvino-findings.md"),
    (DOCS / "03-submission-copy.md", "docs/03-submission-copy.md"),
    (DOCS / "04-demo-video-script.md", "docs/04-demo-video-script.md"),
    (DOCS / "07-judge-simulation-prompt.md", "docs/07-judge-simulation-prompt.md"),
    (ROOT / "out/montage.mp4", "video/montage.mp4"),
    (ROOT / "out/cover.png", "video/cover.png"),
    (ROOT / "out/eval_clips/video/seed000.mp4", "video/one_full_episode_seed000.mp4"),
]
ITEMS += [(p, f"results/{p.name}") for p in sorted((ROOT / "results").glob("*"))
          if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="do not copy; exit non-zero if the packet is stale")
    ap.add_argument("--out", default=str(ROOT / "judging"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    stale, missing, copied = [], [], 0

    for src, rel in ITEMS:
        dst = out / rel
        if not src.exists():
            missing.append(str(src.relative_to(ROOT) if ROOT in src.parents else src))
            continue
        if a.check:
            if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
                stale.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    index = out / "INDEX.md"
    if not index.exists():
        print(f"WARNING: {index} is missing. It is hand-written -- this script does "
              f"not generate it, and the packet is much less useful without it.",
              file=sys.stderr)

    for m in missing:
        print(f"missing (skipped): {m}", file=sys.stderr)

    if a.check:
        if stale:
            print("STALE -- rerun scripts/build_judging_packet.py:", file=sys.stderr)
            for s in stale:
                print(f"  {s}", file=sys.stderr)
            return 1
        print(f"packet is current ({len(ITEMS) - len(missing)} files)")
        return 0

    print(f"wrote {copied} files to {out}")
    print("INDEX.md is hand-written -- check its 'known weaknesses' list still matches "
          "reality before handing the packet to anyone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
