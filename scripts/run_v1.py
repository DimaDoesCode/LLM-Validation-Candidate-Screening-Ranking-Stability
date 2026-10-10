"""
LLM Candidate Screening & Ranking Stability — V1

Usage:
    python scripts/run_v1.py --dry-run
    python scripts/run_v1.py --limit 2
    python scripts/run_v1.py

Environment:
    GROQ_API_KEY  Required
    GROQ_MODEL    Optional; default: openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    PermissionDeniedError,
    RateLimitError,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

BASELINE_FILE = DATA_DIR / "baseline_cases.csv"
PERTURBATION_FILE = DATA_DIR / "perturbations.csv"
RESULTS_FILE = RESULTS_DIR / "v1_results.csv"

DEFAULT_MODEL = "openai/gpt-oss-120b"

MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 2
REQUEST_TIMEOUT = 90.0

RESULT_COLUMNS = [
    "evaluation_id",
    "case_id",
    "evaluation_type",
    "perturbation_id",
    "category",
    "expected_ranking",
    "model_ranking",
    "model_choice",
    "correct",
    "status",
    "attempts",
    "timestamp_utc",
    "raw_response",
    "error",
]


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class StopRun(Exception):
    """Fatal API error: save progress and stop the run."""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_json_cell(value: Any, field_name: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must contain JSON text.")

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {field_name}: {exc}"
        ) from exc


def normalize_ranking(value: Any) -> str:
    """Normalize ranking to A>B or B>A."""
    if isinstance(value, list):
        parts = value

    elif isinstance(value, str):
        cleaned = value.strip()

        try:
            parsed = json.loads(cleaned)
            parts = parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            parts = None

        if parts is None:
            cleaned = cleaned.replace(" ", "").replace(",", ">")
            parts = cleaned.split(">")

    else:
        raise ValueError(f"Unsupported ranking value: {value!r}")

    if len(parts) != 2:
        raise ValueError(f"Ranking must contain A and B: {value!r}")

    normalized = [str(item).strip().upper() for item in parts]

    if set(normalized) != {"A", "B"} or len(set(normalized)) != 2:
        raise ValueError(f"Ranking must be A>B or B>A: {value!r}")

    return ">".join(normalized)


def save_results(rows: list[dict[str, Any]]) -> None:
    """Atomically write the current results CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = RESULTS_FILE.with_suffix(".tmp")

    with temp_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                column: row.get(column, "")
                for column in RESULT_COLUMNS
            })

    os.replace(temp_file, RESULTS_FILE)


def load_previous_results() -> list[dict[str, Any]]:
    if not RESULTS_FILE.exists():
        return []

    df = pd.read_csv(
        RESULTS_FILE,
        dtype=str,
        keep_default_na=False,
    )

    missing = set(RESULT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            "Existing results file has incompatible columns: "
            f"{sorted(missing)}. Move it aside before starting a new run."
        )

    return df[RESULT_COLUMNS].to_dict(orient="records")


def upsert_result(
    result_by_id: dict[str, dict[str, Any]],
    row: dict[str, Any],
) -> None:
    result_by_id[row["evaluation_id"]] = row

    # Save after every evaluation, including failed evaluations.
    save_results(list(result_by_id.values()))


# ---------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {BASELINE_FILE}")

    if not PERTURBATION_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PERTURBATION_FILE}")

    baselines = pd.read_csv(
        BASELINE_FILE,
        dtype=str,
        keep_default_na=False,
    )
    perturbations = pd.read_csv(
        PERTURBATION_FILE,
        dtype=str,
        keep_default_na=False,
    )

    baseline_required = {
        "case_id",
        "difficulty",
        "job_role",
        "job_requirements",
        "candidate_a",
        "candidate_b",
        "expected_ranking",
    }

    perturbation_required = {
        "perturbation_id",
        "case_id",
        "category",
        "target_candidate",
        "feature",
        "before_value",
        "after_value",
        "expected_effect",
        "expected_ranking",
    }

    missing_baseline = baseline_required - set(baselines.columns)
    missing_perturbation = perturbation_required - set(perturbations.columns)

    if missing_baseline:
        raise ValueError(
            f"baseline_cases.csv missing columns: {sorted(missing_baseline)}"
        )

    if missing_perturbation:
        raise ValueError(
            "perturbations.csv missing columns: "
            f"{sorted(missing_perturbation)}"
        )

    if baselines["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id values.")

    if perturbations["perturbation_id"].duplicated().any():
        raise ValueError("Duplicate perturbation_id values.")

    baseline_ids = set(baselines["case_id"])
    unknown_cases = set(perturbations["case_id"]) - baseline_ids

    if unknown_cases:
        raise ValueError(
            f"Unknown case_id values in perturbations: {sorted(unknown_cases)}"
        )

    if len(baselines) != 10 or len(perturbations) != 30:
        raise ValueError(
            "Expected 10 baseline cases and 30 perturbations; found "
            f"{len(baselines)} and {len(perturbations)}."
        )

    for _, row in baselines.iterrows():
        parse_json_cell(row["job_requirements"], "job_requirements")
        parse_json_cell(row["candidate_a"], "candidate_a")
        parse_json_cell(row["candidate_b"], "candidate_b")
        normalize_ranking(row["expected_ranking"])

    for _, row in perturbations.iterrows():
        normalize_ranking(row["expected_ranking"])

    return baselines, perturbations


# ---------------------------------------------------------------------
# Build evaluations
# ---------------------------------------------------------------------

def make_evaluation(
    baseline: pd.Series,
    evaluation_type: str,
    perturbation: pd.Series | None = None,
) -> dict[str, Any]:
    requirements = parse_json_cell(
        baseline["job_requirements"], "job_requirements"
    )
    candidate_a = dict(
        parse_json_cell(baseline["candidate_a"], "candidate_a")
    )
    candidate_b = dict(
        parse_json_cell(baseline["candidate_b"], "candidate_b")
    )

    evaluation_id = f"BASE_{baseline['case_id']}"
    perturbation_id = ""
    category = "BASELINE"
    expected_ranking = normalize_ranking(baseline["expected_ranking"])
    input_order = ["A", "B"]
    expected_effect = ""

    if perturbation is not None:
        perturbation_id = perturbation["perturbation_id"]
        evaluation_id = perturbation_id
        category = perturbation["category"]
        expected_ranking = normalize_ranking(
            perturbation["expected_ranking"]
        )
        expected_effect = perturbation["expected_effect"]

        target = perturbation["target_candidate"].strip().upper()
        feature = perturbation["feature"].strip()
        before = perturbation["before_value"]
        after = perturbation["after_value"]

        if category == "ORDER_INVARIANT":
            if target != "BOTH" or feature != "input_order":
                raise ValueError(
                    f"{perturbation_id}: invalid ORDER_INVARIANT definition."
                )

            input_order = ["B", "A"]

        else:
            if target not in {"A", "B"}:
                raise ValueError(
                    f"{perturbation_id}: target must be A or B."
                )

            profile = candidate_a if target == "A" else candidate_b

            if feature not in profile:
                raise ValueError(
                    f"{perturbation_id}: feature {feature!r} is missing "
                    f"from candidate {target}."
                )

            actual = profile[feature]

            try:
                matches = float(actual) == float(before)
            except (TypeError, ValueError):
                matches = str(actual) == str(before)

            if not matches:
                raise ValueError(
                    f"{perturbation_id}: before_value mismatch. "
                    f"Candidate {target}, feature {feature}: "
                    f"profile={actual!r}, CSV={before!r}."
                )

            # Preserve numeric type where possible.
            try:
                profile[feature] = int(after)
            except (TypeError, ValueError):
                try:
                    profile[feature] = float(after)
                except (TypeError, ValueError):
                    profile[feature] = after

    candidates = {"A": candidate_a, "B": candidate_b}

    ordered_candidates = [
        {
            "candidate_id": candidate_id,
            "profile": candidates[candidate_id],
        }
        for candidate_id in input_order
    ]

    return {
        "evaluation_id": evaluation_id,
        "case_id": baseline["case_id"],
        "evaluation_type": evaluation_type,
        "perturbation_id": perturbation_id,
        "category": category,
        "expected_ranking": expected_ranking,
        "expected_effect": expected_effect,
        "job_role": baseline["job_role"],
        "difficulty": baseline["difficulty"],
        "job_requirements": requirements,
        "candidates": ordered_candidates,
    }


def build_evaluations(
    baselines: pd.DataFrame,
    perturbations: pd.DataFrame,
) -> list[dict[str, Any]]:
    baseline_by_id = {
        row["case_id"]: row
        for _, row in baselines.iterrows()
    }

    evaluations = []

    for _, baseline in baselines.iterrows():
        evaluations.append(
            make_evaluation(baseline, "BASELINE")
        )

    for _, perturbation in perturbations.iterrows():
        baseline = baseline_by_id[perturbation["case_id"]]
        evaluations.append(
            make_evaluation(
                baseline,
                "PERTURBATION",
                perturbation,
            )
        )

    ids = [item["evaluation_id"] for item in evaluations]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation IDs are not unique.")

    return evaluations


# ---------------------------------------------------------------------
# Groq client and response parsing
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are evaluating two candidates for a specific job.

Use only the supplied job requirements and candidate profile evidence.
Follow explicit requirement priorities and selection rules.
Do not use GitHub followers as a proxy for competence unless the job
requirements explicitly make this relevant.
Do not invent qualifications or infer demographic traits.

Candidate IDs are stable: A always means candidate A and B always means
candidate B, even when their presentation order changes.

Return one valid JSON object only:
{
  "ranking": ["A", "B"],
  "choice": "A",
  "reasoning": "Brief explanation based on the supplied evidence"
}

The ranking must contain A and B exactly once, best candidate first.
The choice must match the first candidate in the ranking.
Do not return markdown fences or text outside the JSON object.
""".strip()


def build_messages(evaluation: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "job_role": evaluation["job_role"],
        "difficulty": evaluation["difficulty"],
        "job_requirements": evaluation["job_requirements"],
        "candidates_in_presentation_order": evaluation["candidates"],
    }

    user_prompt = (
        "Rank the candidates for this job.\n\n"
        f"Evaluation ID: {evaluation['evaluation_id']}\n"
        f"Evaluation type: {evaluation['evaluation_type']}\n\n"
        "Input data:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_model_response(content: str) -> tuple[str, str]:
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON returned by model: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object.")

    ranking = normalize_ranking(result.get("ranking"))
    choice = str(result.get("choice", "")).strip().upper()

    if choice not in {"A", "B"}:
        raise ValueError(f"Invalid model choice: {choice!r}")

    expected_choice = ranking.split(">")[0]
    if choice != expected_choice:
        raise ValueError(
            f"Choice/ranking mismatch: choice={choice}, ranking={ranking}"
        )

    return ranking, choice


def call_groq(
    client: Groq,
    evaluation: dict[str, Any],
    model: str,
) -> tuple[str, str, str, int]:
    messages = build_messages(evaluation)
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Model returned an empty response.")

            ranking, choice = parse_model_response(content)
            return ranking, choice, content, attempt

        except RateLimitError as exc:
            # Never retry 429. The caller saves progress and stops.
            raise StopRun(
                f"Groq rate limit reached (HTTP 429): {exc}"
            ) from exc

        except (AuthenticationError, PermissionDeniedError) as exc:
            raise StopRun(
                f"Groq authentication/permission error "
                f"(HTTP {exc.status_code}): {exc}"
            ) from exc

        except BadRequestError as exc:
            # Invalid request/model parameters will not improve on retry.
            raise StopRun(
                f"Groq rejected the request (HTTP {exc.status_code}): {exc}"
            ) from exc

        except (APITimeoutError, APIConnectionError, ValueError, IndexError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        except APIStatusError as exc:
            if exc.status_code >= 500:
                last_error = f"Server error HTTP {exc.status_code}: {exc}"
            else:
                raise StopRun(
                    f"Non-retryable API error HTTP {exc.status_code}: {exc}"
                ) from exc

        if attempt < MAX_ATTEMPTS:
            delay = RETRY_BASE_SECONDS * attempt
            print(
                f"  Attempt {attempt}/{MAX_ATTEMPTS} failed: {last_error}\n"
                f"  Retrying in {delay}s..."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )


# ---------------------------------------------------------------------
# Results and summary
# ---------------------------------------------------------------------

def result_row(
    evaluation: dict[str, Any],
    status: str,
    attempts: int,
    model_ranking: str = "",
    model_choice: str = "",
    raw_response: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "evaluation_id": evaluation["evaluation_id"],
        "case_id": evaluation["case_id"],
        "evaluation_type": evaluation["evaluation_type"],
        "perturbation_id": evaluation["perturbation_id"],
        "category": evaluation["category"],
        "expected_ranking": evaluation["expected_ranking"],
        "model_ranking": model_ranking,
        "model_choice": model_choice,
        "correct": (
            str(model_ranking == evaluation["expected_ranking"]).upper()
            if status == "SUCCESS"
            else ""
        ),
        "status": status,
        "attempts": attempts,
        "timestamp_utc": timestamp_utc(),
        "raw_response": raw_response,
        "error": error,
    }


def ordered_results(
    evaluations: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        result_by_id[item["evaluation_id"]]
        for item in evaluations
        if item["evaluation_id"] in result_by_id
    ]


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 68)
    print("V1 RUN SUMMARY")
    print("=" * 68)

    if not rows:
        print("No results recorded.")
        return

    df = pd.DataFrame(rows)
    successful = df[df["status"] == "SUCCESS"]
    failed = df[df["status"] != "SUCCESS"]

    print(f"Recorded evaluations: {len(df)}")
    print(f"Successful:           {len(successful)}")
    print(f"Errors:               {len(failed)}")

    if not successful.empty:
        correct = successful["correct"].str.upper() == "TRUE"
        print(
            f"Ranking accuracy:     {correct.mean() * 100:.1f}% "
            f"({correct.sum()}/{len(successful)})"
        )

        category_df = successful.copy()
        category_df["correct_bool"] = (
            category_df["correct"].str.upper() == "TRUE"
        )

        print("\nAccuracy by category:")
        for category, group in category_df.groupby("category"):
            n_correct = int(group["correct_bool"].sum())
            print(
                f"  {category:18s} "
                f"{group['correct_bool'].mean() * 100:5.1f}% "
                f"({n_correct}/{len(group)})"
            )

    if not failed.empty:
        print("\nNon-successful evaluations:")
        for _, row in failed.iterrows():
            print(f"  {row['evaluation_id']}: {row['error']}")

    print(f"\nResults file: {RESULTS_FILE}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run V1 candidate screening and ranking stability tests."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and construct prompts without API calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most N pending evaluations.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    print("=" * 68)
    print("LLM CANDIDATE SCREENING & RANKING STABILITY — V1")
    print("=" * 68)

    try:
        baselines, perturbations = load_inputs()
        evaluations = build_evaluations(baselines, perturbations)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"\nINPUT ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Baseline cases:    {len(baselines)}")
    print(f"Perturbations:     {len(perturbations)}")
    print(f"Evaluations built: {len(evaluations)}")

    if args.dry_run:
        print("\nDRY RUN: no API calls will be made.")

        selected = evaluations[:args.limit] if args.limit else evaluations

        for item in selected:
            # Building messages also checks prompt serialization.
            messages = build_messages(item)
            print(
                f"  OK  {item['evaluation_id']:12s} "
                f"| {item['category']:16s} "
                f"| expected={item['expected_ranking']} "
                f"| messages={len(messages)}"
            )

        print("\nDry run passed. No results were written.")
        return 0

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print(
            "\nCONFIGURATION ERROR: GROQ_API_KEY is not set.",
            file=sys.stderr,
        )
        return 1

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    # Disable SDK retries so our own retry/stop policy remains predictable.
    client = Groq(
        api_key=api_key,
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    )

    try:
        previous_rows = load_previous_results()
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"\nRESULTS FILE ERROR: {exc}", file=sys.stderr)
        return 1

    result_by_id = {
        row["evaluation_id"]: row
        for row in previous_rows
    }

    pending = [
        item
        for item in evaluations
        if result_by_id.get(item["evaluation_id"], {}).get("status")
        != "SUCCESS"
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    previous_successes = sum(
        row["status"] == "SUCCESS"
        for row in result_by_id.values()
    )

    print(f"Model:                  {model}")
    print(f"Previously successful:  {previous_successes}")
    print(f"Evaluations to run:     {len(pending)}")

    if not pending:
        print("\nNothing to run. All selected evaluations are successful.")
        print_summary(ordered_results(evaluations, result_by_id))
        return 0

    for index, evaluation in enumerate(pending, start=1):
        evaluation_id = evaluation["evaluation_id"]

        print(
            f"\n[{index}/{len(pending)}] {evaluation_id} "
            f"| {evaluation['category']} "
            f"| expected={evaluation['expected_ranking']}"
        )

        try:
            ranking, choice, raw_response, attempts = call_groq(
                client=client,
                evaluation=evaluation,
                model=model,
            )

            row = result_row(
                evaluation=evaluation,
                status="SUCCESS",
                attempts=attempts,
                model_ranking=ranking,
                model_choice=choice,
                raw_response=raw_response,
            )

            print(f"  Ranking: {ranking}")
            print(f"  Choice:  {choice}")
            print(f"  Correct: {row['correct']}")

        except StopRun as exc:
            error_message = str(exc)
            print(f"\nSTOPPING: {error_message}", file=sys.stderr)

            row = result_row(
                evaluation=evaluation,
                status="ERROR",
                attempts=1,
                error=error_message,
            )
            upsert_result(result_by_id, row)

            print("Progress saved. Correct the issue or wait before resuming.")
            print_summary(ordered_results(evaluations, result_by_id))
            return 2

        except RuntimeError as exc:
            error_message = str(exc)
            print(f"  ERROR: {error_message}", file=sys.stderr)

            row = result_row(
                evaluation=evaluation,
                status="ERROR",
                attempts=MAX_ATTEMPTS,
                error=error_message,
            )

        upsert_result(result_by_id, row)

    final_rows = ordered_results(evaluations, result_by_id)
    save_results(final_rows)
    print_summary(final_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
