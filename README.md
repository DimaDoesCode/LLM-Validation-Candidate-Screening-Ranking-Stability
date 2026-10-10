# LLM Candidate Screening & Ranking Stability

A compact black-box validation experiment for evaluating how an LLM ranks job candidates and responds to controlled changes in candidate profiles.

**V1 status: Completed** · **Evaluations: 40** · **Observed ranking accuracy: 100%**

## Project Objective

Candidate screening is a decision-making task in which an LLM must compare candidate profiles against explicit job requirements.

A model may produce a plausible ranking while still being sensitive to irrelevant information, presentation order, or small changes in candidate characteristics.

This project explores how to validate such behavior using controlled synthetic cases, paired comparisons, and reproducible evaluation scripts.

The project is an experimental validation exercise, not a production hiring system.

## V1 Experiment

V1 evaluates two candidates per case against predefined job requirements and expected rankings.

**Model:** `openai/gpt-oss-120b` via Groq API  
**Temperature:** `0`  
**Evaluation mode:** Black-box, structured JSON output

### Dataset

| Component | Cases |
|---|---:|
| Baseline cases | 10 |
| `IMPROVE` perturbations | 5 |
| `REVERSAL` perturbations | 5 |
| `NO_CHANGE` perturbations | 10 |
| `ORDER_INVARIANT` perturbations | 10 |
| **Total evaluations** | **40** |

The baseline dataset contains five `CLEAR` and five `NEAR_TIE` cases.

Each perturbation is applied independently to its original baseline case.

### Test Categories

- **BASELINE** — evaluate the initial ranking against the predefined expected outcome.
- **IMPROVE** — modify a candidate profile while expecting the original winner to remain on top.
- **REVERSAL** — change a decisive candidate feature so that the expected ranking reverses.
- **NO_CHANGE** — modify a feature intended not to affect the ranking.
- **ORDER_INVARIANT** — reverse the presentation order of candidates while preserving their identities and expected ranking.

## V1 Results

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| BASELINE | 10 | 10 | 100.0% |
| IMPROVE | 5 | 5 | 100.0% |
| REVERSAL | 5 | 5 | 100.0% |
| NO_CHANGE | 10 | 10 | 100.0% |
| ORDER_INVARIANT | 10 | 10 | 100.0% |
| **Overall** | **40** | **40** | **100.0%** |

All 40 evaluations completed successfully, with no API or parsing errors.

The model matched the predefined expected ranking in every case.

### Interpretation

V1 demonstrated that the evaluation pipeline works and that the model handled this initial set of ranking scenarios as expected.

However, the cases were relatively simple. Many contained clear differences in high-priority skills, and the reversal cases introduced substantial changes to decisive features.

Consequently, perfect accuracy does not establish general ranking reliability or robustness.

V1 did not adequately measure decision-boundary sensitivity, repeated-run variability, or the quality of the model's reasoning. The `IMPROVE` category also measured preservation of the expected winner rather than the magnitude of improvement in relative candidate strength.

For the full analysis, see [`V1_validation_report.md`](V1_validation_report.md).

## Repository Structure

```text
.
├── data/
│   ├── baseline_cases.csv
│   └── perturbations.csv
├── scripts/
│   ├── validate_dataset.py
│   └── run_v1.py
├── results/
│   └── v1_results.csv
├── reports/
│   └── V1_validation_report.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Python 3.10 or later is recommended.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Validate the dataset:

```bash
python scripts/validate_dataset.py
```

Run a local dry run without API calls:

```bash
python scripts/run_v1.py --dry-run
```

Run the evaluation:

```bash
python scripts/run_v1.py
```

The evaluation script records results incrementally and supports resuming incomplete runs. A Groq API key must be configured in the environment according to the script's configuration.

## Limitations

- Small, synthetic dataset.
- Expected rankings reflect predefined assumptions about job requirements.
- Limited difficulty in several baseline and reversal cases.
- No repeated-run stability analysis.
- No independent evaluation of explanation quality.
- No assessment of real-world hiring validity or fairness.

The results should not be interpreted as evidence that the model is suitable for autonomous hiring decisions.

## Next Step: V2

V1 is frozen as an initial baseline experiment. The next version will focus on more diagnostic tests, potentially including:

- Sensitivity to small changes near ranking boundaries.
- Resistance to explicitly irrelevant features.
- Consistency when candidates have competing strengths.
- Robustness to semantically equivalent changes in presentation.
- Repeated-run variability.

V2 will be designed and evaluated separately, without retroactively modifying V1 results.

## Scope

This project investigates the behavior of an LLM under controlled evaluation conditions. It does not attempt to automate employment decisions or establish the fairness of a hiring process.
