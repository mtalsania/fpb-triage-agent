"""Send requests to Claude in real time (sync) or through the Batch API, or to a
mock that needs no API key. Every mode returns the same shape so the rest of the
code does not care which one ran."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from . import config as C


def cost_usd(model_key: str, usage: dict, batch: bool) -> float:
    p = C.MODELS[model_key]
    dollars = (usage.get("input_tokens", 0) * p["in"]
               + usage.get("output_tokens", 0) * p["out"]
               + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
               + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]) / 1e6
    return dollars * (C.BATCH_DISCOUNT if batch else 1.0)


def _usage_dict(u) -> dict:
    return {k: int(getattr(u, k, 0) or 0) for k in
            ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}


# ---------------------------------------------------------------- real API --
def _client():
    import anthropic
    return anthropic.Anthropic(max_retries=5)


def run_sync(requests: list[tuple[str, dict]]) -> dict:
    client = _client()

    def one(item):
        cid, params = item
        t0 = time.perf_counter()
        try:
            msg = client.messages.create(**params)
        except Exception as e:  # keep going; record the failure
            return cid, {"message": None, "error": f"{type(e).__name__}: {e}", "latency_s": None}
        return cid, {"message": msg, "latency_s": time.perf_counter() - t0}

    out = {}
    with ThreadPoolExecutor(C.SYNC_WORKERS) as pool:
        for i, (cid, res) in enumerate(pool.map(one, requests), 1):
            out[cid] = res
            print(f"\r  {i}/{len(requests)} done", end="")
    print()
    return out


def run_batch(requests: list[tuple[str, dict]]) -> dict:
    client = _client()
    batch = client.messages.batches.create(
        requests=[{"custom_id": cid, "params": params} for cid, params in requests])
    print(f"  Batch {batch.id} submitted ({len(requests)} requests). Most finish within an hour.")
    t0 = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        rc = batch.request_counts
        print(f"\r  {int(time.time()-t0)//60} min: {rc.succeeded} done, {rc.processing} processing", end="")
        time.sleep(30)
    print(f"\n  Batch ended after {(time.time()-t0)/60:.1f} min")
    out = {}
    for r in client.messages.batches.results(batch.id):
        if r.result.type == "succeeded":
            out[r.custom_id] = {"message": r.result.message, "latency_s": None}
        else:
            out[r.custom_id] = {"message": None, "error": r.result.type, "latency_s": None}
    return out


# -------------------------------------------------------------------- mock --
def _fill(schema: dict, seed: str):
    """Produce schema-valid junk, deterministic per request. For pipeline tests only."""
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][h % len(schema["enum"])]
    if t == "object":
        return {k: _fill(v, seed + k) for k, v in schema["properties"].items()}
    if t == "array":
        item = schema["items"]
        return [_fill(item, seed + "0")] if h % 3 == 0 else []
    if t == "boolean":
        return h % 4 != 0
    return "MOCK OUTPUT. Not a real model response."


def run_mock(requests: list[tuple[str, dict]]) -> dict:
    out = {}
    for cid, params in requests:
        schema = params["output_config"]["format"]["schema"]
        text = json.dumps(_fill(schema, cid + params["model"]))
        n_in = sum(len(b["text"]) for b in params["system"]) // 4
        msg = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=len(params["messages"][0]["content"]) // 4,
                                  output_tokens=300, cache_read_input_tokens=n_in,
                                  cache_creation_input_tokens=0))
        out[cid] = {"message": msg, "latency_s": 0.0}
    return out


def execute(requests: list[tuple[str, dict]], mode: str) -> dict:
    """mode is 'batch', 'sync' or 'mock'. Returns cid -> {message, latency_s, usage}."""
    fn = {"batch": run_batch, "sync": run_sync, "mock": run_mock}[mode]
    res = fn(requests)
    for r in res.values():
        r["usage"] = _usage_dict(r["message"].usage) if r.get("message") else {}
    return res


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
