"""Download one CFPB archive export, clean it, and draw a stratified sample.

Every cleaning step is counted in data/prep_stats.json so the write-up can say
exactly what was dropped and why.
"""
import io
import json
import re
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config as C

# The archive's column naming is not documented, so accept both the bulk-CSV
# headers ("Consumer complaint narrative") and the API field names.
ALIASES = {
    "narrative": ["consumer_complaint_narrative", "complaint_what_happened", "narrative"],
    "product": ["product"],
    "sub_product": ["sub_product", "subproduct"],
    "issue": ["issue"],
    "sub_issue": ["sub_issue", "subissue"],
    "company": ["company"],
    "state": ["state"],
    "tags": ["tags"],
    "company_response": ["company_response_to_consumer", "company_response"],
    "timely": ["timely_response", "timely"],
    "complaint_id": ["complaint_id"],
    "date_received": ["date_received"],
}
NON_FINAL_RESPONSES = {"in progress", ""}


def _snake(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def download(export: str) -> Path:
    name = C.EXPORTS[export]
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = C.RAW_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached {dest.name}")
        return dest
    url = C.ARCHIVE_BASE + name
    print(f"Downloading {url}\n(this can take a few minutes)")
    req = urllib.request.Request(url, headers={"User-Agent": "cfpb-triage-eval/1.0"})
    # Python from python.org on macOS ships without system certificates, which
    # fails with CERTIFICATE_VERIFY_FAILED. certifi's bundle avoids that.
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:,.0f} / {total/1e6:,.0f} MB", end="")
        print()
    except Exception as e:  # network blocked, file renamed, etc.
        dest.unlink(missing_ok=True)
        sys.exit(f"Download failed ({e}).\nDownload it with:\n  mkdir -p {dest.parent} && "
                 f"curl -L -o {dest} {url}\nthen re-run.")
    return dest


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        if member.lower().endswith(".csv"):
            return pd.read_csv(fh, dtype=str, keep_default_na=False, low_memory=False)
        raw = fh.read()
    text = raw.decode("utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:  # JSON lines
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, dict):  # API-style {"hits": {"hits": [...]}}
        obj = obj.get("hits", {}).get("hits", obj)
    rows = [r.get("_source", r) for r in obj]
    return pd.DataFrame(rows).astype(str).replace({"None": "", "nan": ""})


def load_export(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".csv", ".json", ".jsonl"))]
        if not members:
            sys.exit(f"No CSV/JSON inside {path.name}: {zf.namelist()[:10]}")
        df = pd.concat([_read_member(zf, m) for m in members], ignore_index=True)
    df.columns = [_snake(c) for c in df.columns]
    out = {}
    for canon, options in ALIASES.items():
        hit = next((o for o in options if o in df.columns), None)
        out[canon] = df[hit] if hit else ""
        if hit is None and canon in ("narrative", "product", "issue"):
            sys.exit(f"Required column '{canon}' not found. Columns are: {list(df.columns)}")
    return pd.DataFrame(out).fillna("")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_in_export": len(df)}
    df = df[df.narrative.str.len() >= C.MIN_NARRATIVE_CHARS]
    stats["with_narrative_ge_min_chars"] = len(df)
    df = df[~df.company_response.str.strip().str.lower().isin(NON_FINAL_RESPONSES)]
    stats["with_final_company_response"] = len(df)
    # Credit-repair mills submit the same template narrative thousands of times.
    # Keep one copy so the model is not scored on the same text repeatedly.
    key = df.narrative.str.lower().str.replace(r"[^a-z]", "", regex=True)
    stats["duplicate_template_narratives_removed"] = int(key.duplicated().sum())
    df = df[~key.duplicated()].copy()
    stats["after_dedup"] = len(df)
    df["truncated"] = df.narrative.str.len() > C.MAX_NARRATIVE_CHARS
    df["narrative"] = df.narrative.str[: C.MAX_NARRATIVE_CHARS]
    tags = df.tags.str.lower()
    df["tag_servicemember"] = tags.str.contains("servicemember")
    df["tag_older_american"] = tags.str.contains("older american")
    resp = df.company_response.str.lower()
    df["proxy_high_risk"] = resp.str.contains("monetary relief") & ~resp.str.contains("non-monetary")
    df["proxy_high_risk"] |= df.timely.str.lower().eq("no") | resp.eq("untimely response")
    if not df.complaint_id.astype(bool).all():
        df["complaint_id"] = [f"row{i}" for i in range(len(df))]
    return df, stats


def taxonomy(df: pd.DataFrame) -> dict:
    """Products and their issues, from the cleaned pool (not just the sample)."""
    tax = {}
    for prod, g in df.groupby("product"):
        tax[prod] = sorted(i for i in g.issue.unique() if i)
    return dict(sorted(tax.items()))


def stratified_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = df["product"].value_counts()
    keep = counts[counts >= C.MIN_PRODUCT_SUPPORT].index
    if len(keep) == 0:
        sys.exit(f"No product has {C.MIN_PRODUCT_SUPPORT}+ usable complaints. Counts: {counts.to_dict()}")
    parts = [df[df["product"] == p].sample(n=min(C.PER_PRODUCT, counts[p]), random_state=C.SEED)
             for p in keep]
    sample = pd.concat(parts).sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    natural = (counts[keep] / counts[keep].sum()).round(4).to_dict()
    return sample, {"product_counts_in_pool": counts.to_dict(),
                    "products_dropped_low_support": [p for p in counts.index if p not in keep],
                    "natural_product_share": natural}


def prepare(export: str = C.DEFAULT_EXPORT) -> None:
    path = download(export)
    df = load_export(path)
    df, stats = clean(df)
    sample, strat = stratified_sample(df)
    stats.update(strat)
    stats["export"] = C.EXPORTS[export]
    stats["sample_size"] = len(sample)
    stats["sample_proxy_high_risk_rate"] = round(float(sample.proxy_high_risk.mean()), 4)
    stats["sample_truncated"] = int(sample.truncated.sum())
    C.DATA_DIR.mkdir(exist_ok=True)
    sample.to_json(C.DATA_DIR / "sample.jsonl", orient="records", lines=True)
    tax = taxonomy(df[df["product"].isin(sample["product"].unique())])
    (C.DATA_DIR / "taxonomy.json").write_text(json.dumps(tax, indent=2))
    (C.DATA_DIR / "prep_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2, default=str))
    print(f"Sample: {len(sample)} complaints across {sample['product'].nunique()} products -> data/sample.jsonl")


def load_sample() -> pd.DataFrame:
    p = C.DATA_DIR / "sample.jsonl"
    if not p.exists():
        sys.exit("No sample yet. Run: python run.py prepare")
    return pd.read_json(p, lines=True, dtype={"complaint_id": str})


def load_taxonomy() -> dict:
    return json.loads((C.DATA_DIR / "taxonomy.json").read_text())
