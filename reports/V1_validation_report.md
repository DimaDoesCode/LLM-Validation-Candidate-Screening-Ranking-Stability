# V1 Validation Report — Candidate Screening & Ranking Stability

## 1. Objective

V1 was designed as a compact black-box experiment to evaluate whether an LLM ranks two candidates according to explicit job requirements and responds appropriately to controlled profile perturbations.

The experiment focused on five categories:

- `BASELINE` — initial candidate ranking.
- `IMPROVE` — improvement of a candidate's profile.
- `REVERSAL` — changes intended to reverse the expected ranking.
- `NO_CHANGE` — changes intended to preserve the ranking.
- `ORDER_INVARIANT` — reversal of candidate presentation order.

## 2. Experimental Setup

- Model provider: Groq API
- Model: `openai/gpt-oss-120b`
- Temperature: 0
- Evaluation mode: black-box, structured JSON output
- Baseline cases: 10
- Perturbations: 30
- Total evaluations: 40

The baseline dataset contained five `CLEAR` and five `NEAR_TIE` cases. The perturbation dataset contained 5 `IMPROVE`, 5 `REVERSAL`, 10 `NO_CHANGE`, and 10 `ORDER_INVARIANT` cases.

Each perturbation was applied independently to its original baseline case.

## 3. Results

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| BASELINE | 10 | 10 | 100.0% |
| IMPROVE | 5 | 5 | 100.0% |
| REVERSAL | 5 | 5 | 100.0% |
| NO_CHANGE | 10 | 10 | 100.0% |
| ORDER_INVARIANT | 10 | 10 | 100.0% |
| **Overall** | **40** | **40** | **100.0%** |

- Successful evaluations: 40
- API or parsing errors: 0
- Incorrect rankings against predefined expectations: 0

Results were recorded in `results/v1_results.csv`.

## 4. Interpretation

The model reproduced the predefined expected ranking in all 40 evaluations.

The experiment demonstrated that the evaluation pipeline could construct baseline and perturbed candidate profiles, submit them to the model, parse structured responses, record results, and calculate category-level accuracy.

The model also responded correctly to the five deliberately constructed reversal cases and preserved the expected ranking in the ten presentation-order invariance cases.

However, the observed performance should be interpreted within the limits of the experimental design.

### 4.1 Limited challenge difficulty

Many baseline cases contained pronounced differences in high-priority skills. The reversal cases also introduced substantial changes to the decisive feature. These conditions made the intended ranking relatively easy to infer.

### 4.2 Limited diagnostic value of overall accuracy

The evaluation used agreement with predefined expected rankings as its primary metric. This does not fully capture the quality of ranking behavior.

In particular, the `IMPROVE` category measured whether the expected winner remained on top, not whether the improvement produced a meaningful change in relative ranking strength.

### 4.3 Limited evidence about stability

The `NO_CHANGE` tests examined a narrow set of controlled changes. The `ORDER_INVARIANT` tests examined candidate presentation order but did not cover broader paraphrasing, requirement rewording, or other semantically equivalent representations.

Each evaluation was performed once. Therefore, the experiment did not measure run-to-run variability or provide evidence about stochastic reproducibility.

### 4.4 Ranking accuracy does not establish reasoning quality

The model's explanations were recorded, but they were not independently evaluated for completeness, factual support, or consistency with the supplied profiles.

A correct ranking alone does not establish that the model used a valid or reliable decision process.

## 5. Limitations

1. Small synthetic dataset: 10 baseline cases and 30 perturbations.
2. Predefined expected rankings may encode assumptions that are not universally valid hiring criteria.
3. No independent assessment of the substantive correctness of every case and expected outcome.
4. No systematic measurement of decision boundaries or minimum changes required to reverse a ranking.
5. No repeated-run stability analysis.
6. No evaluation of real-world hiring validity, fairness, or job-related predictive performance.

The experiment does not establish that the model is generally reliable for candidate screening or employment decisions.

## 6. Conclusion

V1 achieved 100% agreement with the predefined expected rankings across 40 evaluations.

This is a successful baseline pipeline test, but not strong evidence of robust candidate-screening behavior. The absence of errors is plausibly related to the simplicity and transparency of the constructed cases.

The main finding is methodological: a test suite can produce perfect accuracy while providing limited information about sensitivity, robustness, and decision stability.

V1 is therefore considered complete as an initial baseline experiment. Its results are retained without post hoc modification.

## 7. Direction for V2

V2 should retain the compact black-box approach while improving the diagnostic value of the experiment.

Priority areas:

- Boundary sensitivity: behavior under small changes near a ranking decision boundary.
- Irrelevant-feature resistance: stability when non-job-related or explicitly irrelevant information changes.
- Trade-off consistency: behavior when candidates have competing strengths across requirements of different priorities.
- Presentation robustness: stability under semantically equivalent changes in ordering and wording.
- More diagnostic metrics: separate measures for appropriate reversals, inappropriate flips, invariance, and repeated-run variability.

V2 should be designed to distinguish justified ranking changes from unjustified instability rather than simply increasing the number of test cases.

---

**Status:** V1 completed. Results frozen. Further experiments belong to V2.
