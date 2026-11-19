"""CFPB complaint triage: one entry point.

    python run.py all                 # everything, end to end (needs ANTHROPIC_API_KEY)
    python run.py all --mock          # same pipeline, fake model, no key, no cost

Or step by step:
    python run.py prepare             # download archive export, clean, sample
    python run.py triage              # classify + flag + draft (Batch API, 50% off)
    python run.py latency             # 20 real-time calls per model, for latency
    python run.py judge               # grade reply drafts with the LLM judge
    python run.py review              # write CSVs for your hand review
    python run.py report              # results/report.md
"""
import argparse
import os
import sys

from triage import config as C
from triage import data, pipeline, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "triage", "latency", "judge", "review", "report", "all"])
    ap.add_argument("--models", default=",".join(C.MODELS), help="comma list, e.g. haiku,sonnet")
    ap.add_argument("--mode", default="batch", choices=["batch", "sync"],
                    help="batch is 50%% cheaper but can take up to an hour; sync is immediate")
    ap.add_argument("--limit", type=int, help="only triage the first N sampled complaints (a cheap trial)")
    ap.add_argument("--export", default=C.DEFAULT_EXPORT, choices=list(C.EXPORTS))
    ap.add_argument("--mock", action="store_true", help="fake model: tests the pipeline with no key or cost")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [m for m in models if m not in C.MODELS]
    if bad:
        sys.exit(f"Unknown model(s) {bad}. Choose from {list(C.MODELS)}")
    mode = "mock" if a.mock else a.mode
    if a.mock:
        C.RESULTS_DIR = C.ROOT / "results_mock"
        C.REVIEW_DIR = C.ROOT / "review_mock"
        print("MOCK MODE: random outputs, results go to results_mock/. Numbers mean nothing.")
    elif a.command in ("triage", "latency", "judge", "all") and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (see README), or add --mock to test without one.")

    cmd = a.command
    if cmd in ("prepare", "all") and not (cmd == "all" and (C.DATA_DIR / "sample.jsonl").exists()):
        data.prepare(a.export)
    if cmd in ("triage", "all"):
        pipeline.triage(models, mode, a.limit, a.force)
    if cmd in ("latency", "all"):
        pipeline.latency(models, mock=a.mock, force=a.force)
    if cmd in ("judge", "all"):
        pipeline.run_judge(models, mode=mode, force=a.force)
    if cmd in ("review", "all"):
        pipeline.review(models)
    if cmd in ("report", "all"):
        report.build_report(models)


if __name__ == "__main__":
    main()
