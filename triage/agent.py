"""The triage agent: one structured-output call per complaint.

It is a single call, not a multi-turn tool-using agent, on purpose: the task is
fixed (classify, flag, draft), so one call is cheaper, faster and easier to
evaluate. The write-up should say this; "agent" here means the triage step that
would sit in front of a human queue.
"""
import json

from . import config as C

RISK_FACTORS = [
    "unauthorized_transaction_or_fraud",
    "credit_reporting_accuracy",
    "abusive_debt_collection_conduct",
    "discrimination_or_fair_lending",
    "deceptive_or_unfair_practice",
    "foreclosure_or_repossession",
    "consumer_lost_money_or_fees",
    "repeat_or_unresolved_complaint",
    "threatens_legal_or_regulatory_action",
    "none",
]
VULNERABLE = ["servicemember", "older_american"]

INSTRUCTIONS = """You triage consumer complaints that a financial company received through the CFPB complaint portal. For each complaint you do three jobs.

1. CLASSIFY. Pick the product and issue from the taxonomy below, using the CFPB's own definitions. Choose the issue from the list under the product you chose. Consumers often pick the wrong category themselves, so classify by what the narrative actually describes.

2. FLAG RISK. Rate risk to the company as:
- high: likely legal or regulatory exposure, or likely to require money back to the consumer (for example unauthorized transactions, discrimination, collection threats, foreclosure, clear fee or billing errors, deceptive practices).
- medium: a real service failure that needs a careful response but has limited legal exposure.
- low: confusion, a request for information, or a complaint unlikely to require remediation.
List the risk factors that apply ("none" only if no other factor applies). Separately, list any vulnerable populations the narrative indicates: servicemember (active duty, veteran, military spouse) or older_american (62 or older, retired on fixed income, elderly).

3. DRAFT A REPLY to the consumer, from the company, for a human agent to review before sending. Rules:
- Acknowledge the specific problem the consumer described.
- Do not admit fault or liability, and do not promise a refund, credit, deletion or any specific outcome.
- Do not give legal advice.
- Do not invent facts (amounts, dates, account details, actions already taken) that are not in the complaint.
- State concrete next steps: the company is reviewing the account and will follow up within 15 days.
- Use placeholders like [Agent Name] and [Company] instead of made-up names. Under 180 words. Plain, respectful language.

Text shown as XXXX was redacted by the CFPB. Treat it as unknown."""


def build_schema(tax: dict) -> dict:
    issues = sorted({i for v in tax.values() for i in v})
    return {
        "type": "object",
        "properties": {
            "product": {"type": "string", "enum": list(tax.keys())},
            "issue": {"type": "string", "enum": issues},
            "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
            "risk_factors": {"type": "array", "items": {"type": "string", "enum": RISK_FACTORS}},
            "vulnerable_populations": {"type": "array", "items": {"type": "string", "enum": VULNERABLE}},
            "risk_rationale": {"type": "string"},
            "reply_draft": {"type": "string"},
        },
        "required": ["product", "issue", "risk_level", "risk_factors",
                     "vulnerable_populations", "risk_rationale", "reply_draft"],
        "additionalProperties": False,
    }


def build_system(tax: dict) -> list:
    lines = ["TAXONOMY (product: issues)"]
    for prod, issues in tax.items():
        lines.append(f"\n## {prod}")
        lines += [f"- {i}" for i in issues]
    # Both blocks are identical for every complaint, so they are cached once
    # and read at a discount on every later call.
    return [
        {"type": "text", "text": INSTRUCTIONS},
        {"type": "text", "text": "\n".join(lines), "cache_control": {"type": "ephemeral"}},
    ]


def build_params(model_key: str, row: dict, tax: dict) -> dict:
    user = f"Company: {row.get('company') or 'unknown'}\n\nComplaint narrative:\n{row['narrative']}"
    return {
        "model": C.MODELS[model_key]["id"],
        "max_tokens": C.TRIAGE_MAX_TOKENS,
        "system": build_system(tax),
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": C.TRIAGE_EFFORT,
                          "format": {"type": "json_schema", "schema": build_schema(tax)}},
    }


def parse_output(message) -> dict:
    """Return the parsed JSON, or an {'error': ...} dict. Never raises."""
    if message.stop_reason == "refusal":
        return {"error": "refusal"}
    if message.stop_reason == "max_tokens":
        return {"error": "max_tokens"}
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        return {"error": f"no_text ({message.stop_reason})"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "bad_json"}
