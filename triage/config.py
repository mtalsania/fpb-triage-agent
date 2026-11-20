"""All tunable settings in one place. Edit here, not in the other modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
REVIEW_DIR = ROOT / "review"   # CSVs you fill in by hand

# --- Data source -------------------------------------------------------------
# CFPB stopped publishing complaint narratives on 2026-08-14. The live API now
# returns labels only. The text survives in the FOIA "narratives archive":
# https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/
ARCHIVE_BASE = "https://files.consumerfinance.gov/f/documents/"
EXPORTS = {
    # Export 9 = Jan-Feb 2025: current product taxonomy, and company responses
    # are final (not "In progress"). Change the key to try another window.
    "9": "CCDB_Export_9_January_2025_through_February_2025.zip",
    "10": "CCDB_Export_10_March_2025_through_April_2025.zip",
    "8": "CCDB_Export_8_November_2024_through_December_2024.zip",
}
DEFAULT_EXPORT = "9"

# --- Sampling ----------------------------------------------------------------
SEED = 42
PER_PRODUCT = 35          # complaints sampled per product (stratified)
MIN_PRODUCT_SUPPORT = 35  # drop products with fewer usable complaints than this
MIN_NARRATIVE_CHARS = 200
MAX_NARRATIVE_CHARS = 6000  # truncate very long narratives (cost control); logged
ISSUE_MIN_SUPPORT = 5     # issues with fewer sampled examples are left out of issue macro-F1

# --- Models ------------------------------------------------------------------
# Short name -> API model id. Prices are USD per million tokens, standard tier.
# Verify against https://www.anthropic.com/pricing before quoting in the write-up.
MODELS = {
    "haiku":  {"id": "claude-haiku-5-5",  "in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "sonnet": {"id": "claude-sonnet-5-5", "in": 2.00, "out": 10.0, "cache_read": 0.10, "cache_write": 2.50},
    "opus":   {"id": "claude-opus-5-5",   "in": 4.00, "out": 20.0, "cache_read": 0.20, "cache_write": 5.00},
}
BATCH_DISCOUNT = 0.5      # Message Batches API bills 50% of standard
TRIAGE_EFFORT = "low"     # classification + a short draft does not need deep thinking
TRIAGE_MAX_TOKENS = 4096  # thinking tokens count toward this, so leave headroom

JUDGE_MODEL = "opus"
JUDGE_EFFORT = "medium"
JUDGE_N_PER_MODEL = 40    # replies judged per model
LATENCY_N = 20            # real-time (non-batch) calls per model for latency numbers
SYNC_WORKERS = 4

# Human risk labels (review/risk_labels.csv)
RISK_LABEL_N = 100
RISK_LABEL_MAX_PROXY_POS = 40  # oversample proxy positives so both classes appear
