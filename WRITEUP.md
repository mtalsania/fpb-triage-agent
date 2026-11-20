# Triaging CFPB complaints with Claude: what worked, what it cost, and what I'd change

## The problem

When a consumer complains to the CFPB about a bank, lender or debt collector, the company is expected to respond within 15 days. Before anyone can investigate, someone has to read the complaint and answer three questions: which team owns it, whether legal or compliance needs to see it, and what to say first.

I built a Claude triage step that answers all three, then measured how far each answer can be trusted, what it costs, and how fast it runs, across three model tiers.

## What I built

- **One structured call per complaint.** Claude reads the consumer's narrative and returns the product and issue (chosen from CFPB's own list), a risk level with reasons, vulnerable-consumer flags, and a reply draft for a human to review. I deliberately did not build a multi-step agent: the task is fixed, so a single call is cheaper, faster and easier to evaluate.
- **An evaluation harness** that runs the same 385 complaints through Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 and scores every output.
- **Cost controls:** the Batch API (half price) for the main run, and prompt caching for the instructions and category list, which made up 91% of input tokens.

Code: this repository. Total API spend for the full evaluation was about $5.

## Data, and two premise problems

**1. The narratives had been removed from the live database.** CFPB stopped publishing complaint text on Aug 14, 2026; the public API now returns labels only. I used CFPB's FOIA narratives archive (the Jan–Feb 2025 export): 811,019 complaints, of which 238,331 had a usable narrative.

**2. Almost half the narratives were duplicates.** 116,034 of them (49%) were word-for-word copies, mostly the same consumer filing against all three credit bureaus or credit-repair templates. Keeping them would have scored the model on the same text many times and inflated accuracy. After removing them, I drew 35 complaints from each of 11 products (385 in total).

**3. Only classification has ground truth.** The data has product and issue labels, but no compliance-risk label and no "correct" reply. Instead of pretending otherwise:

| Output | What it's scored against |
|---|---|
| Product, issue | CFPB's labels. Consumers choose these from a dropdown, so I also hand-reviewed 29 disagreements to estimate label noise |
| Risk | An outcome proxy (the company paid money or responded late), **and** my own high/low labels on 100 complaints, made with the outcome and model answers hidden |
| Reply | An LLM judge (Opus) on 7 yes/no compliance checks, validated against my own grades on 20 replies |

## Results

| | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| Product accuracy (balanced sample) | 79.5% | 81.6% | 83.1% |
| Product accuracy (real-world product mix) | 87.7% | 88.5% | 89.2% |
| Issue accuracy | 37.1% | 49.9% | 53.0% |
| Risk F1 vs my labels | 72% | 71% | 71% |
| Replies passing all compliance checks (judge) | 62.5% | 97.5% | 90.0% |
| Cost per 1,000 complaints (batch) | $0.21 | $3.04 | $6.45 |
| Typical response time (real time) | 3.3s | 4.3s | 6.6s |

**Headline: Sonnet is the model I'd deploy.** It routes as well as Opus (the gap is within the ±4-point noise of a 385-complaint sample), its replies scored at least as well, and it costs half as much. Haiku is 31× cheaper than Opus and good enough to route a complaint to a product team. It is clearly worse at the finer issue category (16 points behind), and its drafts are the weakest.

**Label noise.** In 55% of the cases where Opus disagreed with the CFPB label, I judged Opus's answer defensible. Adjusted for that, Opus's real product accuracy is about 92%. Roughly half of its measured "errors" are consumers picking the wrong category.

**Risk.** My labels showed that the outcome proxy is a poor yardstick: it caught only 43% of the complaints I would escalate, with agreement barely above chance (kappa 0.17). Against my labels, the models catch 80–90% of what I'd escalate, but they flag about two-thirds of everything as high risk. A rule that flags every complaint would score an F1 of 68%; the best model scores 72%. As built, the risk flag is a safety net, not a filter. The models also almost never answer "low", so the three-level scale effectively collapsed into one.

**Replies.** Every model addressed the consumer's specific issue, avoided promises and avoided legal advice. The judge agreed with my grades 90–100% of the time on every check. The main difference between models is invented details. When I reviewed the cases where the judge and I disagreed, I found the first version of the judge was too strict in one way and right in another. It penalized "we are reviewing your account", a sentence my own reply policy requires. It also correctly caught a draft that called the consumer's account a "savings account" when the complaint never said so, which I had missed. I fixed the rubric so policy-required language isn't counted, and re-ran the judge: [[haiku / sonnet / opus no_fabricated_facts pass rate after fix]].

## Failure modes

- **Adjacent products blur.** The most common confusions are "debt or credit management" and "debt collection" read as "credit reporting", and "prepaid card" read as "money transfer". These are real overlaps in CFPB's taxonomy (a collection account appears on a credit report), so the fix is multi-label routing, not a bigger model.
- **Over-flagging risk.** A flag that fires on two of every three complaints protects no one; reviewers would learn to ignore it.
- **Vulnerable consumers are mostly invisible in the text.** CFPB tags 61 sampled complaints as servicemember or older American, but only 14–16 narratives say so. All three models catch those same few. This is a ceiling in the data, not a model failure.
- **Redactions.** CFPB replaces names, dates and amounts with XXXX, which removes signal the model would have in production.

## What I'd do in production

1. **Route with Sonnet; use Haiku only where it's proven.** Product routing on Haiku is a defensible cost saving at volume. Issue tagging and reply drafting should stay on Sonnet.
2. **Fix the risk flag before trusting it.** Tighten the definition of "high", add labeled examples to the prompt, and re-test on held-out labels from compliance reviewers rather than a single labeler.
3. **Use model disagreement as an escalation signal.** Sonnet and Opus disagreed on 18 of 100 risk calls. "Two models disagree" is a cheap, explainable trigger for human review.
4. **Never auto-send replies.** Drafts go to an agent queue, and the real quality metric is how much agents edit them before sending.
5. **Re-run this eval on every new CFPB export,** and alert when the category mix shifts.

## Limits

One labeler (me, not a compliance officer) produced the risk labels, the label-noise verdicts and the reply grades. Samples are small: about ±4 points on product accuracy, and about ±10 points on reply checks with 40 replies per model. The risk-label set over-samples complaints that ended in money or a late response, so its rates describe that set, not all complaints.

## Resume bullet

Built and evaluated a Claude-powered triage pipeline on 385 real CFPB complaints (from an 811K-record archive, after removing 49% duplicate narratives) across 11 product lines, benchmarking three model tiers on accuracy, cost and latency. Recommended the mid-tier model: 88.5% routing accuracy on the real-world product mix and 97.5% compliance-safe reply drafts, at half the top tier's cost ($3.04 per 1,000 complaints). Showed that an outcome-based risk proxy missed 57% of complaints a human would escalate.
