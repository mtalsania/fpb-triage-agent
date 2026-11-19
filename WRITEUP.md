# Triaging CFPB complaints with Claude: what worked, what it cost, and what I'd change

> Draft. Every `[[...]]` is a number to copy from `results/report.md` after the real run. Sections marked **(check)** are predictions written before the run: keep them if the data agrees, and rewrite them if it doesn't. A write-up that confirms every prediction it made in advance reads as untested.

## The problem

A financial company that gets CFPB complaints has 15 days to respond, and every complaint needs three things before a person can act: routing (which product team?), a risk read (does legal or compliance need to see it?), and a first draft reply. I built a Claude triage step that does all three, and an evaluation that measures how far each part can be trusted.

## What I built

- A single structured-output call per complaint that returns product, issue, risk level, risk factors, vulnerable-population flags and a draft reply. ([agent.py](triage/agent.py))
- An evaluation harness that compares three Claude models (Haiku 5.5, Sonnet 5.5, Opus 5.5) on quality, cost per 1,000 complaints and latency.
- Batch processing and prompt caching to keep costs low, plus real-time calls to measure the latency an agent would actually wait for.

## Data, and two premise problems

1. **The narratives were removed from the live database.** CFPB stopped publishing complaint text on Aug 14, 2026. I used the FOIA narratives archive, export [[export]], which gives a frozen and citable snapshot. Of [[rows_in_export]] complaints, [[with_narrative]] had usable text. I removed [[dupes]] duplicate template narratives (credit-repair firms file identical letters at volume) and drew a stratified sample of [[sample_size]] across [[n_products]] products.
2. **Only classification has ground truth.** There is no compliance-risk label and no gold reply. Instead of pretending otherwise:
   - Risk is scored against an *outcome proxy*: the complaint ended in monetary relief or a late response. A proxy measures what happened, not what the risk was, so I report lift over the base rate instead of accuracy.
   - Replies are graded by an LLM judge on 7 yes/no checks, and the judge is validated against my own grades on 20 replies.

## Results

| | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| Product accuracy (balanced / real-world mix) | [[ ]] | [[ ]] | [[ ]] |
| Issue accuracy | [[ ]] | [[ ]] | [[ ]] |
| Risk lift over base rate ([[base]] base rate) | [[ ]] | [[ ]] | [[ ]] |
| Replies passing all compliance checks | [[ ]] | [[ ]] | [[ ]] |
| Cost per 1,000 complaints (batch) | [[ ]] | [[ ]] | [[ ]] |
| Latency p50 / p95 | [[ ]] | [[ ]] | [[ ]] |

**The headline:** [[one sentence: which model you'd deploy, and what it costs relative to the best]].

**Label noise.** When I hand-reviewed [[audit_n]] cases where the model and the dataset disagreed, the model's label was defensible [[share]] of the time. Measured accuracy understates real accuracy by roughly [[gap]] points, because the "ground truth" is a consumer's own choice from a dropdown.

**Judge trust.** The judge agreed with me [[range]] of the time per criterion. It was weakest on [[criterion]], so that criterion's pass rate should be read as [[looser/stricter]] than reality.

## Failure modes (check)

- **Adjacent products blur.** I expect most errors between credit reporting and debt collection (a collection account on a credit report fits both), and between checking/savings and money transfer. These are routing ambiguities, not model errors: the fix is a multi-label taxonomy, not a better model.
- **The risk proxy is weak by construction.** Monetary relief depends on what the company decided to do, which no narrative can predict. Low precision against the proxy does not mean the flags are wrong. Hand-labeling 100 complaints is the next step if this number matters.
- **Vulnerable-population recall is capped.** Tags are set by the consumer on the form, and many narratives never mention military service or age. Recall below [[x]] is partly a ceiling, not only a miss.
- **Reply drafts drift toward promises.** "We will correct this" is a commitment. Watch the `no_unauthorized_commitment` rate. It is the check most likely to matter to a compliance team.
- **Redactions.** CFPB replaces names, dates and amounts with XXXX, which removes signal the model would have in production.

## What I'd do in production

1. **Route on confidence, not on every call.** Send complaints the cheap model is unsure about (or where two cheap runs disagree) to the larger model, and send high-risk flags to a person. The measured cost gap is what makes this worth building.
2. **Never auto-send replies.** Drafts go to an agent queue. Track the edit distance between draft and sent reply as the real quality metric.
3. **Replace the proxy.** Get 300–500 complaints labeled for risk by the compliance team. That becomes the actual acceptance test.
4. **Monitor drift.** CFPB has changed its taxonomy before (most visibly in 2017). Re-run this eval on each new export, and alert when the label distribution shifts.
5. **Data handling.** Real complaints contain PII before redaction. Production would need a data-retention agreement, PII handling, and an audit log of every model decision.

## Resume bullet

[[paste the bullet generated at the end of results/report.md, then tighten it]]
