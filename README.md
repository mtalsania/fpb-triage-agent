# CFPB complaint triage agent, with an honest eval

A Claude-powered triage step for consumer complaints from the CFPB Consumer Complaint Database. For each complaint it:

1. **Classifies** the product and issue, using the CFPB's own taxonomy.
2. **Flags compliance risk** (high / medium / low, risk factors, and vulnerable populations such as servicemembers and older Americans).
3. **Drafts a reply** for a human agent to review.

The evaluation harness compares Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 on quality, cost and latency. That cost-versus-quality tradeoff is the decision a real deployment has to make.

## What can and can't be measured (read this first)

| Output | Ground truth in the data? | How it's scored |
|---|---|---|
| Product, issue | Yes, but noisy: consumers pick the category themselves | Accuracy and macro-F1 against dataset labels, plus a hand audit of disagreements to estimate label noise |
| Compliance risk | **No.** The dataset has no risk label | Against an **outcome proxy**: the company gave monetary relief or responded late. Reported as lift over the base rate. Also checks recall of the consumer-reported Servicemember / Older American tags |
| Reply draft | **No.** There are no gold replies | An LLM judge (Opus 5.5) grades 7 yes/no rubric checks. The judge is checked against your own grades on 20 replies |

**Data source note.** CFPB stopped publishing complaint narratives on Aug 14, 2026, and the live API now returns labels only. The text survives in CFPB's [FOIA narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). This project uses one frozen export (Jan–Feb 2025 by default). That window uses a single product taxonomy, and its company responses are final.

## Setup (about 10 minutes, Mac)

```bash
# 1. Python 3.11 or newer. Check with: python3 --version
cd cfpb-triage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Test the whole pipeline offline. This needs no key and costs nothing.
python -m pytest -q

# 3. API key from https://console.anthropic.com/settings/keys
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run it

```bash
python run.py prepare                                             # downloads the export, cleans, samples ~350 complaints
python run.py triage --models haiku --mode sync --limit 10 --force # $0.01 trial: catches setup problems fast
python run.py all                                                 # the real run, batch mode
```

`all` runs triage (Batch API, 50% cheaper, usually done within an hour), 20 real-time calls per model for latency, the reply judge, the review CSVs and the report. Results are cached on disk, so re-running never pays twice unless you add `--force`. (Delete `results/triage_haiku.jsonl` after the trial, or the main run will reuse those 10 rows.)

**Expected cost** for the full run with all three models: about $5–10. Most of that is Opus and the judge. Haiku alone costs pennies.

Then do the two hand reviews (about 1 hour in total):

```bash
python run.py review
# review/disagreements_<model>.csv : for each case, is label A or B right? (A, B, both, neither)
# review/reply_grading.csv         : 1 or 0 per rubric criterion, for 20 replies
python run.py report               # re-scores with your reviews included
```

The report is `results/report.md`. It ends with a draft resume bullet filled in from your actual numbers.

## How it's built

```
run.py              one command line entry point
triage/config.py    every knob: models, prices, sample sizes, effort
triage/data.py      download, clean (dedupe template narratives), stratified sample
triage/agent.py     prompt + JSON schema for one structured-output call
triage/runner.py    Batch API, real-time and mock execution; cost accounting
triage/judge.py     reply rubric for the LLM judge
triage/pipeline.py  the steps; results cached in results/
triage/report.py    metrics, confusion matrices, report, resume bullet
tests/              offline end-to-end test with fake data and a mock model
```

Design choices worth knowing:

- **One call, not a tool-using loop.** The task is fixed, so one structured-output call is cheaper, faster and easier to evaluate than a multi-step agent.
- **Structured outputs** (`output_config.format`) constrain product and issue to the real taxonomy, so every response parses.
- **Prompt caching.** The instructions and taxonomy are identical for every complaint, so they are cached once and then read at a fraction of the price. The report shows the cache hit share.
- **Batch API** for the main run (50% off). Real-time calls only for latency.
- **Low effort** for triage, since classification doesn't need deep reasoning. You can change it in `config.py` as an experiment.
- **Stratified sample.** CFPB data is dominated by credit reporting, so the sample takes an equal number per product. The report shows accuracy both ways: balanced, and reweighted to the real-world mix.
- **Duplicate template narratives removed.** Credit-repair firms file the same letter thousands of times. Leaving those in would inflate accuracy.
