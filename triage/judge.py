"""LLM-as-judge for reply drafts.

The dataset has no gold replies, so drafts are graded against a fixed rubric of
yes/no checks. Known limits, to state in the write-up:
- The judge is a Claude model grading Claude output, and it grades its own
  family's drafts (Opus judging Opus), so self-preference bias is possible.
- Yes/no checks capture policy compliance better than quality.
- The judge is only trusted as far as it agrees with the human grades in
  review/reply_grading.csv (see `python run.py review`).
"""
from . import config as C

CRITERIA = {
    "addresses_specific_issue": "The reply acknowledges the specific problem the consumer described, not a generic apology.",
    "no_liability_admission": "The reply does not admit fault, wrongdoing or legal liability.",
    "no_unauthorized_commitment": "The reply does not promise a refund, credit, fee reversal, deletion or any specific outcome.",
    "no_legal_advice": "The reply does not give legal advice or interpret the consumer's legal rights.",
    "no_fabricated_facts": ("The reply invents no amounts, dates, account types, account details or past actions "
                            "that are absent from the complaint. Saying the company is reviewing the account and "
                            "will follow up within 15 days is required by the reply policy and is not fabrication."),
    "clear_next_steps": "The reply states concrete next steps and a timeframe.",
    "professional_tone": "The reply is respectful, plain and free of jargon or defensiveness.",
}
# A reply failing any of these would need rewriting before it could be sent.
COMPLIANCE_CRITICAL = ["no_liability_admission", "no_unauthorized_commitment",
                       "no_legal_advice", "no_fabricated_facts"]

SYSTEM = ("You audit draft replies that a financial company plans to send to consumers who "
          "complained through the CFPB. Grade the draft strictly against each criterion. "
          "Answer true only if the draft clearly meets it. Be strict: when in doubt, false.\n\n"
          + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items()))

SCHEMA = {
    "type": "object",
    "properties": {**{k: {"type": "boolean"} for k in CRITERIA}, "notes": {"type": "string"}},
    "required": list(CRITERIA) + ["notes"],
    "additionalProperties": False,
}


def build_params(narrative: str, reply: str) -> dict:
    return {
        "model": C.MODELS[C.JUDGE_MODEL]["id"],
        "max_tokens": 4096,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content":
                      f"<complaint>\n{narrative}\n</complaint>\n\n<draft_reply>\n{reply}\n</draft_reply>"}],
        "output_config": {"effort": C.JUDGE_EFFORT,
                          "format": {"type": "json_schema", "schema": SCHEMA}},
    }
