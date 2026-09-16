#!/usr/bin/env python3
"""Fail the build if any retracted claim has crept back into the copy.

Two review rounds have now scored this submission down for the same failure mode:
a sentence that was true of an earlier draft, corrected in one file, and left
standing in another. The README says the VLM's plan is rejected on every seed; the
cover image said "VLM planner on OpenVINO"; the video said "the plan you just saw
was written by the model". Nobody lied — the copy drifted, and drift is a class of
bug, so it gets a test rather than a resolution to be careful.

Each rule below is a phrase that is FALSE in this project, paired with the file
that retracts it and the wording to use instead. Add a rule the moment a review
catches a claim; never delete one, because the wording it bans is exactly the
wording that will feel natural again in a hurry.

    python scripts/audit_claims.py             # audit repo + packet + hackathon-docs
    python scripts/audit_claims.py --quiet     # exit code only, for a pre-commit hook

Exit code 0 = clean, 1 = a retracted claim is present, 2 = bad invocation.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / "hackathon-docs"

SEARCH_ROOTS = [ROOT / "README.md", ROOT / "docs", ROOT / "judging", ROOT / "scripts",
                ROOT / "results", DOCS]
SUFFIXES = {".md", ".py", ".txt", ".json"}

# Files that are ALLOWED to contain a banned phrase, because their job is to name
# it: this script, and the notes that record what was corrected.
ALLOW = {"audit_claims.py"}

# A line may opt out by carrying this marker -- for the editing notes that have to
# quote a banned phrase in order to ban it ("do not restore X"). The marker is
# deliberately ugly and has to be typed on purpose: an opt-out that is easy to add
# absent-mindedly is the same as no audit at all.
ESCAPE = "audit-allow"

# A block escape, for a table whose whole job is to quote the retracted phrasings
# and say what replaced them. Between these two markers the audit is off.
BLOCK_ON = "audit-allow-block"
BLOCK_OFF = "/audit-allow-block"


class Rule:
    def __init__(self, name: str, pattern: str, why: str, instead: str,
                 allow: set[str] | None = None):
        self.name = name
        self.rx = re.compile(pattern, re.I)
        self.why = why
        self.instead = instead
        self.allow = allow or set()


RULES = [
    Rule("spoon-handoff",
         r"hand(?:ing|s)?\s+the\s+spoon\s+across|the\s+spoon\s+between\s+them",
         "The hand-off is the plate on all 100 seeds, never the spoon. "
         "tests/test_handoff_is_emergent.py pins this down.",
         "hand the PLATE across / the plate between them"),

    Rule("cover-vlm-planner",
         r"VLM planner on OpenVINO",
         "This was the cover subtitle. The VLM's plan is rejected on every seed, "
         "so the 100/100 beside it belongs to the deterministic fallback.",
         "two SO-101 arms . plans validated against physics . 100/100 seeds",
         # bench_planner.py is a benchmark OF the VLM planner on OpenVINO; the
         # phrase is its subject, not a claim about what drove the episodes.
         allow={"bench_planner.py"}),

    Rule("model-wrote-the-plan",
         r"plan you just saw was written by the model"
         r"|the model planned (?:this|the) episode"
         r"|planned by a VLM(?! on OpenVINO proposes)",
         "False on every committed seed. The README retracts it in its first "
         "qualification, so a video or caption saying it contradicts the repo.",
         "'The model wrote a plan. The validator rejected it. The fallback "
         "finished the job.'"),

    Rule("only-looked-at-ten",
         r"only ever looked at the first ten"
         r"|we only looked at seeds 0.9"
         r"|never looked at the (?:other )?(?:ninety|90)",
         "Retracted in the README: three fixes were chosen by counting failures "
         "across all 100 seeds.",
         "'nothing was tuned per seed'"),

    Rule("real-two-arm-concurrency",
         r"real two-arm concurrency (?:happens|is)"
         r"|genuine two-arm concurrency (?:happens|is)"
         r"|concurrency (?:happens |is )?(?:inside|within) the hand-?off",
         "The hand-off is a place-and-pick through a shared cell: arm A withdraws "
         "before arm B approaches. Only the pour has both arms moving at once.",
         "'the hand-off is a place-and-pick through a shared cell; the pour is the "
         "one simultaneous two-arm action'"),

    Rule("openvino-detector-in-the-loop",
         r"detector \(OpenVINO, INT8\)"
         r"|detectors? compiled to OpenVINO IR ground",
         "The OpenVINO INT8 detector is exported and benchmarked but never called "
         "at run time; an HSV ColourDetector closes the loop. docs/06 measures why.",
         "name the HSV colour detector as the in-loop one, and mark the INT8 "
         "detector 'exported and benchmarked, not in the loop'"),

    Rule("npu-driver-missing",
         r"NPU driver is missing(?! ; see)|usually means the NPU\s+\"?driver is missing",
         "The host is an AMD Ryzen AI 9 HX 370. It has no Intel NPU at all, so "
         "there is no driver to be missing.",
         "'the host is AMD; no Intel NPU exists, and none is expected'"),

    Rule("2b-reaches-scheduler-unscoped",
         r"A 2B planner reaches the scheduler\s+reliably"
         r"|2B planner reaches the scheduler reliably",
         "True of the INT8 bench build only. The INT4 build that ships is rejected "
         "at the validator on every seed.",
         "scope it: 'at INT8 -- the bench configuration, not the shipped one --'"),

    Rule("three-qualifications",
         r"Three qualifications, stated here",
         "There are four now: the fourth is that 100/100 is a screened "
         "distribution at a stated tolerance.",
         "'Four qualifications, stated here'"),
]


def files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in SEARCH_ROOTS:
        if root.is_file():
            out.append(root)
        elif root.is_dir():
            out += [p for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUFFIXES]
    return sorted(set(out))


TEST_COUNT_RE = re.compile(r"\b(\d{2,4})\s+(?:tests|passed)\b")


def check_test_count_agrees(quiet: bool) -> list[str]:
    """Every file that states a test count must state the same one.

    "Test count wrong in three places" was a finding in the first review. The
    number moves whenever anyone adds a test, and the copy does not move with it,
    so the failure recurs by default. This does not know the true count -- it
    knows that four files disagreeing is always a bug, which is the part that
    actually got scored.
    """
    # Only the documents that SHIP. The older planning docs (01-project-spec,
    # 02-build-plan) record what the suite looked like when they were written and
    # are history, not claims -- forcing them to track the current count would be
    # rewriting a diary.
    shipped = {"README.md", "INDEX.md", "03-submission-copy.md",
               "04-demo-video-script.md", "06-openvino-findings.md"}
    found: dict[str, list[str]] = {}
    for p in files():
        if p.name not in shipped or p.suffix.lower() != ".md":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in TEST_COUNT_RE.finditer(text):
            found.setdefault(m.group(1), []).append(p.name)

    if len(found) <= 1:
        if not quiet and found:
            n = next(iter(found))
            print(f"test count: every file says {n} "
                  f"({len(set(sum(found.values(), [])))} file(s) agree)")
        return []
    return [f"test count disagrees across files: "
            + "; ".join(f"{n} in {sorted(set(fs))}" for n, fs in sorted(found.items()))
            + " -- run `python -m pytest -q` and make them all match"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    hits: list[tuple[Rule, pathlib.Path, int, str]] = []
    checked = 0
    for p in files():
        if p.name in ALLOW:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        checked += 1
        muted = False
        for i, line in enumerate(text.splitlines(), 1):
            if BLOCK_OFF in line:
                muted = False
                continue
            if BLOCK_ON in line:
                muted = True
                continue
            if muted or ESCAPE in line:
                continue
            for rule in RULES:
                if p.name in rule.allow:
                    continue
                if rule.rx.search(line):
                    hits.append((rule, p, i, line.strip()))

    if not a.quiet:
        print(f"audited {checked} files against {len(RULES)} retracted claims")

    count_problems = check_test_count_agrees(a.quiet)
    for problem in count_problems:
        print(f"\n  [test-count] {problem}", file=sys.stderr)

    if not hits:
        if not a.quiet and not count_problems:
            print("clean -- no retracted claim is present")
        return 1 if count_problems else 0

    if not a.quiet:
        by_rule: dict[str, list] = {}
        for rule, p, i, line in hits:
            by_rule.setdefault(rule.name, []).append((rule, p, i, line))
        print(f"\n{len(hits)} occurrence(s) of {len(by_rule)} retracted claim(s):\n",
              file=sys.stderr)
        for name, group in by_rule.items():
            rule = group[0][0]
            print(f"  [{name}] {rule.why}", file=sys.stderr)
            print(f"      say instead: {rule.instead}", file=sys.stderr)
            for _, p, i, line in group:
                try:
                    shown = p.relative_to(ROOT.parent)
                except ValueError:
                    shown = p
                print(f"      {shown}:{i}: {line[:120]}", file=sys.stderr)
            print(file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
