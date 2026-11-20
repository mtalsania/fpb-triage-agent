"""Turn saved results into results/report.md and results/summary.json.

Every number in the write-up and the resume bullet should come from here."""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from . import config as C
from .data import load_sample
from .judge import COMPLIANCE_CRITICAL, CRITERIA
from .runner import load_jsonl


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.1f}%"


def _load(kind, m):
    p = C.RESULTS_DIR / f"{kind}_{m}.jsonl"
    return pd.DataFrame(load_jsonl(p)) if p.exists() else None


def score_model(m, gold, stats):
    pred = _load("triage", m)
    if pred is None:
        return None
    df = gold.merge(pred, on="cid", how="left", suffixes=("", "_pred"))
    ok = df["product_pred"].notna() if "product_pred" in df else pd.Series(False, index=df.index)
    d = df[ok]
    r = {"model": m, "model_id": C.MODELS[m]["id"], "n": len(df), "errors": int((~ok).sum())}
    # Errors count as wrong: a triage step that fails still sends work to a human.
    correct_p = (df["product"] == df["product_pred"]) & ok
    correct_i = (df["issue"] == df["issue_pred"]) & ok
    r["product_acc"] = correct_p.mean()
    r["product_macro_f1"] = f1_score(df["product"], df["product_pred"].fillna("ERROR"), average="macro",
                                     labels=sorted(df["product"].unique()), zero_division=0)
    share = stats.get("natural_product_share", {})
    per_prod = correct_p.groupby(df["product"]).mean()
    r["product_acc_natural_mix"] = float(sum(share.get(p, 0) * a for p, a in per_prod.items())
                                         / max(sum(share.get(p, 0) for p in per_prod.index), 1e-9))
    r["issue_acc"] = correct_i.mean()
    r["issue_acc_given_product_correct"] = correct_i[correct_p].mean() if correct_p.any() else None
    common = [i for i, c in Counter(df["issue"]).items() if c >= C.ISSUE_MIN_SUPPORT]
    r["issue_macro_f1_common"] = f1_score(df["issue"], df["issue_pred"].fillna("ERROR"), average="macro",
                                          labels=common, zero_division=0)
    r["n_common_issues"] = len(common)

    # Risk vs the outcome proxy (monetary relief or late response).
    base = df["proxy_high_risk"].mean()
    hi = d["risk_level"].eq("high")
    pos = d["proxy_high_risk"].astype(bool)
    r["risk_proxy_base_rate"] = base
    r["risk_flagged_high_rate"] = hi.mean()
    r["risk_precision"] = pos[hi].mean() if hi.any() else None
    r["risk_recall"] = hi[pos].mean() if pos.any() else None
    r["risk_lift"] = (r["risk_precision"] / base) if r["risk_precision"] is not None and base else None

    # Vulnerable-population tags (consumer-reported at submission).
    vp = d["vulnerable_populations"].apply(lambda x: x if isinstance(x, list) else [])
    for tag, col in (("servicemember", "tag_servicemember"), ("older_american", "tag_older_american")):
        t = d[col].astype(bool)
        flagged = vp.apply(lambda v: tag in v)
        r[f"{tag}_n_tagged"] = int(t.sum())
        r[f"{tag}_recall"] = flagged[t].mean() if t.any() else None
        r[f"{tag}_false_flag_rate"] = flagged[~t].mean() if (~t).any() else None

    # Cost and latency.
    r["cost_total_usd"] = float(pred["cost_usd"].sum())
    r["cost_per_1k_usd"] = 1000 * r["cost_total_usd"] / max(len(pred), 1)
    u = pd.DataFrame(list(pred["usage"]))
    if len(u):
        r["avg_output_tokens"] = float(u.get("output_tokens", pd.Series([0])).mean())
        tot_in = u.sum().get("input_tokens", 0) + u.sum().get("cache_read_input_tokens", 0) + u.sum().get("cache_creation_input_tokens", 0)
        r["cache_hit_share"] = (u.sum().get("cache_read_input_tokens", 0) / tot_in) if tot_in else None
    lat = _load("latency", m)
    if lat is not None and lat["latency_s"].notna().any():
        r["latency_p50_s"] = float(lat["latency_s"].median())
        r["latency_p95_s"] = float(lat["latency_s"].quantile(0.95))
        r["cost_per_1k_realtime_usd"] = 1000 * float(lat["cost_usd"].mean())

    # Reply quality from the judge.
    j = _load("judge", m)
    if j is not None:
        j = j[[c for c in CRITERIA if c in j]].dropna()
        if len(j):
            r["judge_n"] = len(j)
            r["reply_criteria_pass"] = {c: float(j[c].mean()) for c in CRITERIA}
            r["reply_all_pass"] = float(j.all(axis=1).mean())
            r["reply_compliance_pass"] = float(j[COMPLIANCE_CRITICAL].all(axis=1).mean())

    # Confusion matrix for product, saved for the write-up.
    labels = sorted(df["product"].unique())
    cm = confusion_matrix(df["product"], df["product_pred"].fillna("ERROR"), labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(C.RESULTS_DIR / f"confusion_product_{m}.csv")
    pairs = Counter(zip(d["product"], d["product_pred"]))
    r["top_confusions"] = [(a, b, n) for (a, b), n in pairs.most_common() if a != b][:5]

    # Human label-noise audit, if filled in.
    rv = C.REVIEW_DIR / f"disagreements_{m}.csv"
    if rv.exists():
        a = pd.read_csv(rv).dropna(subset=["verdict"])
        a = a[a["verdict"].astype(str).str.strip() != ""]
        if len(a):
            share_ok = a.apply(_defensible, axis=1).mean()
            r["audit_n"] = len(a)
            r["audit_model_defensible_share"] = float(share_ok)
            r["product_acc_noise_adjusted"] = float(r["product_acc"] + (1 - r["product_acc"]) * share_ok)
    return r


def _defensible(row) -> bool:
    v = str(row["verdict"]).strip().lower()
    if v == "both":
        return True
    if v in ("a", "b"):
        return row["_A_is"] == ("model" if v == "a" else "dataset")
    return False


def judge_agreement():
    p = C.REVIEW_DIR / "reply_grading.csv"
    if not p.exists():
        return None
    h = pd.read_csv(p)
    h = h.dropna(subset=list(CRITERIA))
    if h.empty:
        return None
    rows = []
    for m in h["_model"].unique():
        j = _load("judge", m)
        if j is not None:
            rows.append(h[h["_model"] == m].merge(j, on="cid", suffixes=("_human", "_judge")))
    if not rows:
        return None
    x = pd.concat(rows)
    out = {"n": len(x)}
    for c in CRITERIA:
        hu, ju = x[f"{c}_human"].astype(int), x[f"{c}_judge"].astype(int)
        kappa = cohen_kappa_score(hu, ju) if hu.nunique() > 1 or ju.nunique() > 1 else None
        out[c] = {"agreement": float((hu == ju).mean()), "kappa": kappa}
    return out


def _resume_bullet(rows, stats):
    rows = [r for r in rows if r]
    if len(rows) < 1:
        return "(run the eval first)"
    best = max(rows, key=lambda r: r["product_acc"])
    cheap = min(rows, key=lambda r: r["cost_per_1k_usd"])
    s = (f"Built and evaluated a Claude-based triage agent for CFPB consumer complaints "
         f"(classification, compliance-risk flagging, reply drafting) on {best['n']} "
         f"real complaints across {len(stats.get('natural_product_share', {}))} product lines; "
         f"reached {_pct(best['product_acc'])} product accuracy ({best['model']})")
    if cheap["model"] != best["model"]:
        gap = 100 * (best["product_acc"] - cheap["product_acc"])
        ratio = best["cost_per_1k_usd"] / max(cheap["cost_per_1k_usd"], 1e-9)
        rel = f"comes within {gap:.1f} points" if gap > 0.05 else "matches it"
        s += (f", and showed {cheap['model']} {rel} at "
              f"{ratio:.0f}x lower cost (${cheap['cost_per_1k_usd']:.2f} per 1,000 complaints)")
    if best.get("reply_compliance_pass") is not None:
        s += f"; {_pct(best['reply_compliance_pass'])} of judged reply drafts passed all compliance checks"
    return s + "."


def build_report(models):
    gold = load_sample()
    gold["cid"] = "c" + gold["complaint_id"].astype(str)
    stats = json.loads((C.DATA_DIR / "prep_stats.json").read_text())
    rows = [score_model(m, gold, stats) for m in models]
    rows = [r for r in rows if r]
    if not rows:
        print("No results yet. Run: python run.py triage")
        return
    agree = judge_agreement()

    L = ["# CFPB complaint triage: evaluation report", ""]
    L += [f"Data: `{stats['export']}`. {stats['rows_in_export']:,} complaints in the export; "
          f"{stats['with_narrative_ge_min_chars']:,} with a narrative of {C.MIN_NARRATIVE_CHARS}+ characters; "
          f"{stats['duplicate_template_narratives_removed']:,} duplicate template narratives removed; "
          f"{stats['sample_size']} sampled ({C.PER_PRODUCT} per product, stratified).", ""]
    L += ["## Model comparison", "",
          "| Metric | " + " | ".join(r["model"] for r in rows) + " |",
          "|---|" + "---|" * len(rows)]

    def row(label, key, fmt=_pct):
        L.append(f"| {label} | " + " | ".join(fmt(r.get(key)) for r in rows) + " |")
    money = lambda v: "n/a" if v is None else f"${v:,.2f}"
    secs = lambda v: "n/a" if v is None else f"{v:.1f}s"
    num = lambda v: "n/a" if v is None else f"{v:.2f}"
    row("Product accuracy (balanced sample)", "product_acc")
    row("Product accuracy (real-world mix)", "product_acc_natural_mix")
    row("Product macro-F1", "product_macro_f1")
    row("Issue accuracy", "issue_acc")
    row("Issue accuracy when product right", "issue_acc_given_product_correct")
    row("Issue macro-F1 (common issues)", "issue_macro_f1_common")
    row("Risk: flagged high", "risk_flagged_high_rate")
    row("Risk: precision vs proxy", "risk_precision")
    row("Risk: recall vs proxy", "risk_recall")
    row("Risk: lift over base rate", "risk_lift", num)
    row("Servicemember tag recall", "servicemember_recall")
    row("Older American tag recall", "older_american_recall")
    row("Replies passing all compliance checks", "reply_compliance_pass")
    row("Replies passing every check", "reply_all_pass")
    row("Cost per 1,000 complaints (batch)", "cost_per_1k_usd", money)
    row("Cost per 1,000 complaints (real time)", "cost_per_1k_realtime_usd", money)
    row("Latency p50 (real time)", "latency_p50_s", secs)
    row("Latency p95 (real time)", "latency_p95_s", secs)
    row("Share of input tokens from cache", "cache_hit_share")
    row("Failed calls", "errors", lambda v: str(v))
    L += ["", f"Risk proxy base rate (monetary relief or late response): {_pct(rows[0]['risk_proxy_base_rate'])}. "
          "This is an outcome proxy, not a compliance label: it says what happened, not what the risk was.", ""]

    L += ["## Most common product confusions", ""]
    for r in rows:
        L.append(f"**{r['model']}**: " + ("; ".join(f"{a} -> {b} ({n})" for a, b, n in r["top_confusions"]) or "none"))
        L.append("")
    if any("audit_n" in r for r in rows):
        L += ["## Label-noise audit (human review of disagreements)", ""]
        for r in rows:
            if "audit_n" in r:
                L.append(f"- {r['model']}: in {r['audit_n']} reviewed disagreements, the model's label was "
                         f"defensible {_pct(r['audit_model_defensible_share'])} of the time. "
                         f"Noise-adjusted product accuracy is about {_pct(r['product_acc_noise_adjusted'])}.")
        L.append("")
    L += ["## Reply rubric pass rates (LLM judge)", ""]
    jr = [r for r in rows if "reply_criteria_pass" in r]
    if jr:
        L += ["| Criterion | " + " | ".join(r["model"] for r in jr) + " |", "|---|" + "---|" * len(jr)]
        for c in CRITERIA:
            L.append(f"| {c} | " + " | ".join(_pct(r["reply_criteria_pass"][c]) for r in jr) + " |")
    if agree:
        L += ["", f"Judge vs human agreement ({agree['n']} replies):", ""]
        for c in CRITERIA:
            k = agree[c]["kappa"]
            L.append(f"- {c}: {_pct(agree[c]['agreement'])} agreement"
                     + (f", kappa {k:.2f}" if k is not None else ""))
    else:
        L += ["", "Judge not yet checked against human grades. Run `python run.py review` and fill in review/reply_grading.csv."]
    L += ["", "## Draft resume bullet (numbers filled from this run)", "", _resume_bullet(rows, stats), ""]

    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "report.md").write_text("\n".join(L))
    (C.RESULTS_DIR / "summary.json").write_text(json.dumps({"models": rows, "judge_agreement": agree},
                                                           indent=2, default=str))
    print("\n".join(L))
    print("\nSaved results/report.md and results/summary.json")
