# V2 Experimental Design — Candidate Screening & Ranking Stability

## 1. Purpose

V2 is a compact black-box validation experiment for studying how an LLM ranks two job candidates when decision-relevant and irrelevant aspects of the input are controlled.

The experiment is designed to test ranking behavior, not to automate hiring or establish real-world hiring validity.

**Primary research question**

> Does an LLM ranking respond appropriately to controlled, decision-relevant changes while remaining invariant to irrelevant changes and semantically equivalent presentations, and how reproducible is the result across repeated calls?

V1 is complete and remains unchanged. Its 100% observed ranking accuracy applied only to its 40 evaluations on a small, relatively straightforward synthetic test set. V2 addresses some of those limitations; it does not retroactively alter or invalidate V1.

## 2. Scope and model configuration

Planned configuration:

- Provider: Groq API
- Model: `openai/gpt-oss-120b`
- Temperature: `0`
- Client: official Groq Python SDK
- Data handling: Python and pandas
- Output: structured JSON identifying the preferred candidate and, where explicitly allowed, a tie
- Target budget: approximately 50 successful evaluation calls, excluding any additional retries

The exact model identifier, request parameters, run date, and relevant SDK version must be recorded with the results. Temperature 0 does not guarantee identical outputs across all repeated requests or service conditions.

## 3. Hypotheses

### H1 — Boundary sensitivity

When a controlled change crosses a pre-specified decision boundary, the model should change its preference in the expected direction. When a change does not cross that boundary, the preference should remain unchanged.

### H2 — Irrelevant-feature resistance

Changing a feature explicitly defined as irrelevant to the vacancy criteria should not change the ranking.

### H3 — Presentation robustness

Changing the order of candidate presentation or paraphrasing requirements without changing their meaning should not change the ranking.

### H4 — Repeated-run stability

Repeated calls with identical inputs and fixed request settings should produce the same normalized ranking at a high observed rate.

These hypotheses concern different properties. Ranking correctness, response to perturbations, and repeatability must be reported separately; they must not be collapsed into one overall accuracy score.

## 4. Experimental design and planned budget

The initial design targets 12 unique baseline scenarios and approximately 50 API calls. Some baseline scenarios may be reused across paired tests; a reused output must not be counted as multiple independent observations.

| Category | Baseline scenarios | Planned evaluations | Planned calls |
|---|---:|---|---:|
| A. Boundary sensitivity | 4 | Baselines and controlled changes across or near a decision boundary | 16 |
| B. Irrelevant-feature resistance | 3 | Baselines and irrelevant-feature perturbations | 9 |
| C. Presentation robustness | 3 | Baselines, candidate-order swaps, and meaning-preserving paraphrases | 9 |
| D. Repeated-run stability | 2 | Two baselines, each evaluated once plus seven repeat calls | 16 |
| **Total** | **12 unique baselines** | | **50** |

The category counts above are a planning allocation, not a claim that all observations are independent. Before data collection, the case inventory must state exactly which baseline call is shared by which tests so that the final call count is auditable and duplicate evaluations are not double-counted.

### A. Boundary sensitivity — 16 planned calls

Use four scenarios with controlled changes to a decision-relevant feature. The set should include:

- changes that should preserve the preferred candidate;
- changes that should justify a reversal;
- a mandatory-requirement scenario with an explicit rule;
- one or two trade-off scenarios in which the vacancy explicitly states the priority between competing qualifications.

A tie is allowed only in cases deliberately constructed to have a tie under the pre-specified rules. The exact mapping of baseline and perturbation calls must be settled in the case inventory before execution.

### B. Irrelevant-feature resistance — 9 planned calls

Use three baselines and two perturbation checks per baseline. Change only a feature that the scenario specification explicitly declares irrelevant to the job criteria. Avoid introducing sensitive personal attributes; fairness testing is outside this experiment's scope.

### C. Presentation robustness — 9 planned calls

Use three baselines with paired presentation changes:

- candidate-order swaps;
- requirement paraphrases that preserve meaning, constraints, and priority.

Report the two subtypes separately. A paraphrase must be reviewed before API evaluation to confirm that it does not alter the task's meaning.

### D. Repeated-run stability — 16 planned calls

Select two baseline scenarios: one with a clear preference and one with a close comparison. For each, make one initial call and seven additional calls with identical input and settings:

`2 scenarios × (1 initial call + 7 repeats) = 16 calls`

The 14 additional calls are repeats beyond the two initial calls. This small repeated-run sample is a diagnostic check, not a robust estimate of general model variability.

## 5. Case construction and ground-truth rules

Every case and perturbation must be reviewed before any model calls. Expected outcomes must not be adjusted after seeing model responses.

For each case, record at least:

- `case_id`: unique scenario identifier;
- `pair_id`: identifier linking a baseline and its perturbation;
- `category`: A, B, C, or D;
- `baseline_expected`: expected baseline winner or permitted tie;
- `expected_effect`: `KEEP`, `REVERSE`, or `TIE`, where applicable;
- `perturbation_type`: controlled change being tested;
- `decisive_feature`: feature or rule that determines the expected result;
- `justification`: concise, pre-specified rationale;
- the complete vacancy requirements and candidate profiles;
- a record of human review and any revisions made before freezing the dataset.

### Expected effects

- **KEEP:** the preferred candidate must remain the same.
- **REVERSE:** the preferred candidate must switch to the other candidate.
- **TIE:** the case's pre-specified decision rules do not distinguish the candidates, and no other allowed feature breaks the tie.

A case must be revised or excluded if its expected outcome cannot be justified without subjective assumptions not encoded in the scenario.

### Boundary and trade-off rules

A numerical change alone does not establish that a decision boundary has been crossed. The scenario must define why the change changes the expected outcome. For example, a mandatory requirement may be specified as a hard constraint, or the vacancy may explicitly prioritize one qualification over another.

For trade-offs, specify the priority rule in the case itself before evaluation. Do not infer weights from the model's explanation or choose a winner after seeing the output.

### Paired comparisons

Each perturbation should differ from its baseline only in the intended factor. All other candidate information, vacancy criteria, and presentation details should remain fixed unless the test specifically concerns presentation.

A baseline and its perturbation form a paired observation. Report pair-level results; do not treat multiple variants of one underlying scenario as fully independent evidence.

## 6. Metrics

Report category-specific metrics with numerator and denominator. Do not report a single pooled accuracy as the primary result.

### A. Boundary sensitivity

- **Correct effect rate:** fraction of valid perturbations whose observed effect matches the pre-specified `KEEP`, `REVERSE`, or `TIE`.
- **Missed reversal rate:** fraction of required reversals where the original preference remains.
- **Premature reversal rate:** fraction of `KEEP` controls where the model switches preference.
- **Tie error rate:** fraction of cases with an incorrect tie or a failure to return a required tie.

Report the counts for each expected-effect class separately. The correct-effect rate alone can hide failures on a less frequent class.

### B. Irrelevant-feature resistance

- **Ranking invariance rate:** fraction of valid perturbation pairs with unchanged normalized ranking.
- **Unjustified flip rate:** fraction of valid pairs where an irrelevant change changes the ranking.

These are complementary views of the same paired behavior; show their denominators and any invalid or failed calls.

### C. Presentation robustness

Report separately for candidate-order swaps and requirement paraphrases:

- **Invariance rate:** fraction of pairs whose normalized ranking is unchanged.
- Number and direction of observed changes.

Normalize outcomes by candidate ID, not by the displayed A/B position. A correct order-invariance test must not mistake a label or position swap for a ranking change.

### D. Repeated-run stability

- **Exact agreement rate:** successful repeated calls matching the initial normalized ranking, divided by successful repeated calls.
- **Observed flip rate:** successful repeats whose ranking differs from the initial ranking, divided by successful repeated calls.
- **Number of unique normalized rankings** observed for each scenario.
- API and parsing failure counts, reported separately.

Failed calls are not rankings and must not be counted as agreement or as flips. If the initial call fails, the scenario cannot be evaluated for repeat agreement until a valid initial result is obtained.

## 7. Output schema and normalization

The response schema should minimally identify:

- the selected candidate ID, or an explicit `TIE` outcome when the case permits it;
- any required ranking fields defined consistently for all cases.

Normalize the response into a stable internal representation before scoring. Ignore JSON key order and harmless formatting differences. Validate that returned candidate IDs are valid and that the response satisfies the schema. A parse failure or invalid candidate ID is an execution/format error, not a correct ranking.

Textual explanations may be retained for qualitative inspection. They are not proof that the model actually used the feature it mentions and are not a substitute for outcome-based metrics.

## 8. Execution protocol

1. Build and manually review the complete synthetic case inventory.
2. Freeze expected outcomes and the evaluation rules before model calls.
3. Run local dataset validation and a no-API dry run.
4. Run a limited API test to verify the request and response schema.
5. Execute the planned evaluations, saving every successful result immediately.
6. Support resume: skip evaluations already marked successful and retry failed evaluations without overwriting successful outputs.
7. On HTTP 429, save current results and stop immediately.
8. Keep raw response, normalized outcome, status, error details, timestamp, and relevant model/request metadata where available.
9. Do not silently change cases, expected labels, prompts, or scoring rules mid-run. If a substantive correction is necessary, document it, version the dataset/protocol, and distinguish pre-run corrections from post-run changes.

The approximately 50-call budget is a target for planned successful evaluations. Retries may increase total HTTP requests. If the run is interrupted by a rate limit, report the completed subset and do not describe the experiment as complete.

## 9. Interpretation and reporting

Use descriptive outcome labels rather than an arbitrary pass threshold:

- **Expected behavior observed:** all valid tested scenarios show their pre-specified effects.
- **Diagnostic deviation observed:** at least one valid case shows a specific deviation, such as a missed reversal, premature reversal, irrelevant-feature flip, presentation-dependent result, or repeated-run disagreement.
- **Inconclusive:** ambiguity in the case specification, invalid execution, or insufficient completed observations prevents a defensible interpretation.

A 100% result is informative only as a description of behavior on the tested cases. It does not demonstrate general reliability. Always report counts and denominators; uncertainty is substantial with small samples, and related perturbations are not independent observations.

A stable but incorrect baseline is still incorrect. A model that changes its ranking appropriately in one category may still be presentation-sensitive or unstable in another. Discuss these dimensions independently.

## 10. Limitations

- The scenarios are synthetic and cover only a small, deliberately selected set of situations.
- Human-authored expected outcomes can encode assumptions and must be reviewed before evaluation.
- The experiment compares pairs of candidates; it does not establish performance for larger candidate pools or full recruitment workflows.
- Temperature 0 does not guarantee deterministic service behavior.
- The repeated-run sample is too small to characterize general stochastic variability precisely.
- The experiment does not establish hiring validity, job performance prediction, fairness, legal compliance, or suitability for automated employment decisions.
- Model explanations do not establish causal feature use.

## 11. References

These references motivate the general focus on controlled evaluation, robustness, and reproducibility. They do not validate this particular synthetic test set.

- [Ribeiro et al. — CheckList: Beyond Accuracy, Behavioral Testing of NLP Models (ACL 2020)](https://aclanthology.org/2020.acl-main.442/)
- [Sagawa et al. — An Investigation of Why Overparameterization Exacerbates Spurious Correlations (ICML 2020)](https://proceedings.mlr.press/v119/sagawa20a.html)
- [NIST — AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Mitchell et al. — Model Cards for Model Reporting (FAT* 2019)](https://doi.org/10.1145/3287560.3287596)

## 12. Freeze status

This document defines the proposed V2 protocol before case construction and API evaluation. The next step is to draft the case inventory, review every expected outcome, and validate the dataset locally. No API evaluation should begin until the cases, expected effects, and scoring rules are frozen.
