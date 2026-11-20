"""The steps behind each `python run.py <command>`. Results are saved to disk and
reused, so re-running a command never pays for the same calls twice unless you
pass --force."""
import random

import pandas as pd

from . import agent, judge
from . import config as C
from .data import load_sample, load_taxonomy
from .runner import cost_usd, execute, load_jsonl, save_jsonl


def _cid(row) -> str:
    return f"c{row['complaint_id']}"


def _triage_rows(model, df, tax, mode):
    reqs = [(_cid(r), agent.build_params(model, r, tax)) for r in df.to_dict("records")]
    res = execute(reqs, mode)
    rows = []
    for cid, _ in reqs:
        r = res.get(cid, {"message": None, "error": "missing", "usage": {}, "latency_s": None})
        out = agent.parse_output(r["message"]) if r.get("message") else {"error": r.get("error")}
        rows.append({"cid": cid, "mode": mode, "latency_s": r["latency_s"], "usage": r["usage"],
                     "cost_usd": cost_usd(model, r["usage"], batch=(mode == "batch")), **out})
    return rows


def triage(models, mode="batch", limit=None, force=False):
    df, tax = load_sample(), load_taxonomy()
    if limit:
        df = df.head(limit)
    for m in models:
        path = C.RESULTS_DIR / f"triage_{m}.jsonl"
        if path.exists() and not force:
            print(f"[{m}] {path.name} exists, skipping (use --force to re-run)")
            continue
        print(f"[{m}] triaging {len(df)} complaints via {mode}")
        rows = _triage_rows(m, df, tax, mode)
        save_jsonl(path, rows)
        errs = sum(1 for r in rows if "error" in r)
        print(f"[{m}] saved {len(rows)} rows, {errs} errors, ${sum(r['cost_usd'] for r in rows):.3f}")


def latency(models, n=C.LATENCY_N, mock=False, force=False):
    """Real-time calls on a small slice, to measure what an agent waiting on the
    answer would experience. The batch run cannot measure this."""
    df, tax = load_sample().head(n), load_taxonomy()
    for m in models:
        path = C.RESULTS_DIR / f"latency_{m}.jsonl"
        if path.exists() and not force:
            print(f"[{m}] {path.name} exists, skipping")
            continue
        print(f"[{m}] {n} real-time calls for latency")
        save_jsonl(path, _triage_rows(m, df, tax, "mock" if mock else "sync"))


def run_judge(models, n=C.JUDGE_N_PER_MODEL, mode="batch", force=False):
    sample = load_sample()
    sample = sample.set_index(sample.apply(_cid, axis=1))
    for m in models:
        path = C.RESULTS_DIR / f"judge_{m}.jsonl"
        if path.exists() and not force:
            print(f"[{m}] {path.name} exists, skipping")
            continue
        preds = [r for r in load_jsonl(C.RESULTS_DIR / f"triage_{m}.jsonl") if "reply_draft" in r]
        random.Random(C.SEED).shuffle(preds)
        preds = preds[:n]
        reqs = [(p["cid"], judge.build_params(sample.loc[p["cid"], "narrative"], p["reply_draft"]))
                for p in preds]
        print(f"[{m}] judging {len(reqs)} replies with {C.JUDGE_MODEL}")
        res = execute(reqs, mode)
        rows = []
        for cid, _ in reqs:
            r = res[cid]
            out = agent.parse_output(r["message"]) if r.get("message") else {"error": r.get("error")}
            rows.append({"cid": cid, "usage": r["usage"],
                         "cost_usd": cost_usd(C.JUDGE_MODEL, r["usage"], batch=(mode == "batch")), **out})
        save_jsonl(path, rows)


def review(models, n_disagree=30, n_replies=20):
    """Write CSVs for a human to fill in. Filled files feed back into the report."""
    C.REVIEW_DIR.mkdir(exist_ok=True)
    sample = load_sample()
    sample["cid"] = sample.apply(_cid, axis=1)
    s = sample.set_index("cid")
    rng = random.Random(C.SEED)

    # 1) Label-noise audit: where the model and the dataset disagree on product,
    #    who is right? Model names are hidden to avoid bias.
    for m in models:
        out = C.REVIEW_DIR / f"disagreements_{m}.csv"
        if out.exists():
            print(f"{out.name} exists, not overwriting your edits")
            continue
        tp = C.RESULTS_DIR / f"triage_{m}.jsonl"
        if not tp.exists():
            print(f"[{m}] no triage results yet, skipping its disagreement sheet")
            continue
        preds = [r for r in load_jsonl(tp) if "product" in r]
        dis = [p for p in preds if p["product"] != s.loc[p["cid"], "product"]]
        rng.shuffle(dis)
        rows = []
        for p in dis[:n_disagree]:
            a, b = ("dataset", "model") if rng.random() < 0.5 else ("model", "dataset")
            lab = {"dataset": s.loc[p["cid"], "product"], "model": p["product"]}
            rows.append({"cid": p["cid"], "narrative": s.loc[p["cid"], "narrative"],
                         "label_A": lab[a], "label_B": lab[b], "_A_is": a,
                         "verdict": "", "_options": "A / B / both / neither"})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Wrote {out} ({len(rows)} rows). Fill 'verdict' with A, B, both or neither.")

    risk_labels()

    # 3) Reply grading, to check how far the LLM judge can be trusted.
    out = C.REVIEW_DIR / "reply_grading.csv"
    if out.exists():
        print(f"{out.name} exists, not overwriting your edits")
        return
    pool = []
    for m in models:
        p = C.RESULTS_DIR / f"judge_{m}.jsonl"
        if not p.exists():
            continue
        judged = {r["cid"] for r in load_jsonl(p) if "error" not in r}
        for r in load_jsonl(C.RESULTS_DIR / f"triage_{m}.jsonl"):
            if r["cid"] in judged and "reply_draft" in r:
                pool.append((m, r))
    rng.shuffle(pool)
    rows = []
    for m, r in pool[:n_replies]:
        rows.append({"cid": r["cid"], "_model": m, "narrative": s.loc[r["cid"], "narrative"],
                     "reply_draft": r["reply_draft"], **{k: "" for k in judge.CRITERIA}})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows). Put 1 (meets) or 0 (fails) in each criterion column. "
          "Don't look at the _model column until you're done.")

def risk_labels():
    """Write review/risk_labels.csv. Needs only the sample, not any model results."""
    C.REVIEW_DIR.mkdir(exist_ok=True)
    sample = load_sample()
    sample["cid"] = sample.apply(_cid, axis=1)
    # The outcome proxy is thin (~9% positive), so every
    #    proxy-positive complaint is included (up to 40) and the rest is random.
    #    Nothing about the proxy or any model's answer is shown to the labeler.
    out = C.REVIEW_DIR / "risk_labels.csv"
    if out.exists():
        print(f"{out.name} exists, not overwriting your edits")
    else:
        pos = sample[sample["proxy_high_risk"].astype(bool)].sample(frac=1, random_state=C.SEED).head(C.RISK_LABEL_MAX_PROXY_POS)
        rest = sample.drop(pos.index).sample(frac=1, random_state=C.SEED).head(C.RISK_LABEL_N - len(pos))
        pick = pd.concat([pos, rest]).sample(frac=1, random_state=C.SEED)
        pd.DataFrame({"cid": pick["cid"], "company": pick["company"], "narrative": pick["narrative"],
                      "risk": "", "notes": ""}).to_csv(out, index=False)
        print(f"Wrote {out} ({len(pick)} rows). Put 'high' or 'low' in the risk column. "
              "See 'Labeling risk' in the README for what counts as high.")

