#!/usr/bin/env python3
"""Is the exported tokenizer round-tripping JSON correctly?

The exported model emits JSON with stray quotes and doubled colons -- '"skill":":
"pick"' -- which looks like quantisation damage and is easy to blame on INT4. But
transformers warns on load that this checkpoint's tokenizer has "an incorrect
regex pattern" that "will lead to incorrect tokenization", and a tokenizer that
cannot round-trip punctuation would produce exactly that. This checks, rather
than assuming either way.
"""
from __future__ import annotations

import argparse
import sys

SAMPLE = ('{"nodes": [{"id": "n0", "skill": "open_drawer", "args": {}, "arm": "A", '
          '"deps": [], "rationale": "the drawer is closed"}, '
          '{"id": "n1", "skill": "pick", "args": {"object": "plate"}, "arm": "A", '
          '"deps": ["n0"], "rationale": "only arm A reaches it"}]}')


def check(name: str, tok) -> bool:
    ids = tok(SAMPLE, return_tensors=None)["input_ids"]
    back = tok.decode(ids, skip_special_tokens=True)
    ok = back == SAMPLE
    print(f"{name:34s} {'ROUND-TRIPS' if ok else 'CORRUPTS'}  ({len(ids)} tokens)")
    if not ok:
        for i, (a, b) in enumerate(zip(SAMPLE, back)):
            if a != b:
                print(f"    first difference at char {i}:")
                print(f"      sent : ...{SAMPLE[max(0, i - 40):i + 40]!r}")
                print(f"      back : ...{back[max(0, i - 40):i + 40]!r}")
                break
        else:
            print(f"    lengths differ: sent {len(SAMPLE)}, back {len(back)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exported", default="models/qwen2-vl-2b-ov-int8")
    ap.add_argument("--source", default="Qwen/Qwen2-VL-2B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    results = {}
    for label, path, kw in (
        ("exported dir", args.exported, {}),
        ("exported dir, fix_mistral_regex", args.exported, {"fix_mistral_regex": True}),
        ("original HF checkpoint", args.source, {}),
    ):
        try:
            tok = AutoTokenizer.from_pretrained(path, **kw)
        except Exception as exc:                                       # noqa: BLE001
            print(f"{label:34s} could not load: {type(exc).__name__}: {exc}")
            continue
        results[label] = check(label, tok)

    print("\nverdict:")
    if results.get("exported dir") is False and any(
            v for k, v in results.items() if k != "exported dir"):
        print("  the EXPORTED tokenizer is the problem, not the quantisation.")
    elif all(results.values()) and results:
        print("  every tokenizer round-trips; the malformed JSON comes from the "
              "model, so it is a quality/quantisation issue after all.")
    else:
        print("  inconclusive -- see above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
