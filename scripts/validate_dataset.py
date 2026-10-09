#!/usr/bin/env python3
"""Validate the V1 Candidate Screening & Ranking Stability CSV dataset.

Usage:
    python validate_dataset.py
    python validate_dataset.py --data-dir path/to/csvs

Requires: pandas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASELINE_COLUMNS = {
    "case_id", "difficulty", "job_role", "job_requirements",
    "candidate_a", "candidate_b", "primary_differentiator", "expected_ranking",
}
PERTURBATION_COLUMNS = {
    "perturbation_id", "case_id", "category", "target_candidate", "feature",
    "before_value", "after_value", "expected_effect", "expected_ranking",
}
PROFILE_FEATURES = {
    "python_years", "ml_years", "sql_years", "cloud_years",
    "domain_years", "leadership_years", "github_followers",
}
RANKINGS = {"A>B", "B>A"}
CATEGORIES = {"IMPROVE", "REVERSAL", "NO_CHANGE", "ORDER_INVARIANT"}
EXPECTED_COUNTS = {
    "IMPROVE": 5, "REVERSAL": 5, "NO_CHANGE": 10, "ORDER_INVARIANT": 10,
}


def fail(message: str) -> None:
    raise ValueError(message)


def parse_json_cell(value, label):
    try:
        result = json.loads(value)
    except Exception as exc:
        fail(f"{label}: invalid JSON ({exc})")
    if not isinstance(result, dict):
        fail(f"{label}: expected a JSON object")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    baseline_file = data_dir / "baseline_cases.csv"
    perturb_file = data_dir / "perturbations.csv"

    for path in (baseline_file, perturb_file):
        if not path.exists():
            fail(f"Missing file: {path}")

    baseline = pd.read_csv(baseline_file, dtype=str, keep_default_na=False)
    perturb = pd.read_csv(perturb_file, dtype=str, keep_default_na=False)

    missing_b = BASELINE_COLUMNS - set(baseline.columns)
    missing_p = PERTURBATION_COLUMNS - set(perturb.columns)
    if missing_b:
        fail(f"baseline_cases.csv missing columns: {sorted(missing_b)}")
    if missing_p:
        fail(f"perturbations.csv missing columns: {sorted(missing_p)}")

    if len(baseline) != 10:
        fail(f"Expected 10 baseline rows, found {len(baseline)}")
    if len(perturb) != 30:
        fail(f"Expected 30 perturbation rows, found {len(perturb)}")
    if baseline["case_id"].duplicated().any():
        fail("Duplicate case_id in baseline_cases.csv")
    if perturb["perturbation_id"].duplicated().any():
        fail("Duplicate perturbation_id in perturbations.csv")
    if set(baseline["expected_ranking"]) - RANKINGS:
        fail("Invalid baseline expected_ranking value")
    if set(perturb["expected_ranking"]) - RANKINGS:
        fail("Invalid perturbation expected_ranking value")
    if set(baseline["difficulty"]) != {"CLEAR", "NEAR_TIE"}:
        fail("Unexpected difficulty values")

    baseline_by_id = {}
    for _, row in baseline.iterrows():
        req = parse_json_cell(row["job_requirements"], f'{row["case_id"]}.job_requirements')
        a = parse_json_cell(row["candidate_a"], f'{row["case_id"]}.candidate_a')
        b = parse_json_cell(row["candidate_b"], f'{row["case_id"]}.candidate_b')
        if set(a) != PROFILE_FEATURES or set(b) != PROFILE_FEATURES:
            fail(f'{row["case_id"]}: candidate profile keys do not match required feature set')
        for name, profile in (("A", a), ("B", b)):
            for feature in PROFILE_FEATURES:
                try:
                    number = float(profile[feature])
                except (TypeError, ValueError):
                    fail(f'{row["case_id"]}: {name}.{feature} must be numeric')
                if number < 0:
                    fail(f'{row["case_id"]}: {name}.{feature} must be non-negative')
        primary = req.get("primary_requirement")
        if primary not in {"Python", "ML", "SQL", "Cloud", "Domain", "Leadership"}:
            fail(f'{row["case_id"]}: invalid primary_requirement')
        if not row["primary_differentiator"].strip():
            fail(f'{row["case_id"]}: primary_differentiator is empty')
        baseline_by_id[row["case_id"]] = {
            "row": row, "requirements": req, "A": a, "B": b
        }

    unknown_cases = set(perturb["case_id"]) - set(baseline_by_id)
    if unknown_cases:
        fail(f"Perturbations reference unknown cases: {sorted(unknown_cases)}")
    counts = perturb["category"].value_counts().to_dict()
    if set(counts) != CATEGORIES or counts != EXPECTED_COUNTS:
        fail(f"Category counts mismatch. Expected {EXPECTED_COUNTS}, found {counts}")

    for _, row in perturb.iterrows():
        pid = row["perturbation_id"]
        case = baseline_by_id[row["case_id"]]
        category = row["category"]
        if category not in CATEGORIES:
            fail(f"{pid}: invalid category {category}")
        if row["expected_effect"] == "":
            fail(f"{pid}: expected_effect is empty")

        if category in {"IMPROVE", "REVERSAL"}:
            if row["target_candidate"] not in {"A", "B"}:
                fail(f"{pid}: target_candidate must be A or B")
            if row["feature"] not in PROFILE_FEATURES - {"github_followers"}:
                fail(f"{pid}: invalid relevant feature {row['feature']}")
            original = case[row["target_candidate"]][row["feature"]]
            try:
                before, after = float(row["before_value"]), float(row["after_value"])
            except ValueError:
                fail(f"{pid}: before_value and after_value must be numeric")
            if before != float(original):
                fail(f"{pid}: before_value {before} does not match baseline {original}")
            if after <= before:
                fail(f"{pid}: relevant perturbation must increase the feature")
            if category == "IMPROVE" and row["expected_effect"] != "IMPROVE_NO_WORSE":
                fail(f"{pid}: unexpected IMPROVE expected_effect")
            if category == "REVERSAL" and row["expected_effect"] != "REVERSE_WINNER":
                fail(f"{pid}: unexpected REVERSAL expected_effect")
        elif category == "NO_CHANGE":
            if row["target_candidate"] not in {"A", "B"}:
                fail(f"{pid}: NO_CHANGE target_candidate must be A or B")
            if row["feature"] != "github_followers":
                fail(f"{pid}: NO_CHANGE must use github_followers")
            original = case[row["target_candidate"]]["github_followers"]
            if float(row["before_value"]) != float(original):
                fail(f"{pid}: before_value does not match baseline github_followers")
            if float(row["after_value"]) == float(row["before_value"]):
                fail(f"{pid}: NO_CHANGE perturbation must actually change the value")
            if row["expected_effect"] != "PRESERVE_RANKING":
                fail(f"{pid}: unexpected NO_CHANGE expected_effect")
        elif category == "ORDER_INVARIANT":
            if row["target_candidate"] != "BOTH" or row["feature"] != "input_order":
                fail(f"{pid}: invalid order perturbation target/feature")
            if (row["before_value"], row["after_value"]) != ("A,B", "B,A"):
                fail(f"{pid}: order perturbation must swap A,B to B,A")
            if row["expected_effect"] != "PRESERVE_CANDIDATE_PREFERENCE":
                fail(f"{pid}: unexpected ORDER_INVARIANT expected_effect")

        expected_baseline = case["row"]["expected_ranking"]
        if category in {"IMPROVE", "NO_CHANGE", "ORDER_INVARIANT"}:
            if row["expected_ranking"] != expected_baseline:
                fail(f"{pid}: expected ranking should match baseline {expected_baseline}")
        else:
            if row["expected_ranking"] == expected_baseline:
                fail(f"{pid}: REVERSAL expected_ranking must differ from baseline")

    # Expected category/case coverage and output summary.
    improve_cases = set(perturb.loc[perturb["category"] == "IMPROVE", "case_id"])
    reversal_cases = set(perturb.loc[perturb["category"] == "REVERSAL", "case_id"])
    if len(improve_cases) != 5 or len(reversal_cases) != 5:
        fail("Expected IMPROVE and REVERSAL variants to cover five distinct cases each")

    print("DATASET VALIDATION: PASS")
    print(f"Baseline rows:      {len(baseline)}")
    print(f"Perturbation rows:  {len(perturb)}")
    print("Category counts:    " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))
    print("Unique case IDs:    OK")
    print("Unique variant IDs: OK")
    print("JSON profiles:      OK")
    print("Feature/schema:     OK")
    print("Before-values:      OK")
    print("Expected rankings:  OK")
    print("Note: this checks structural consistency, not whether a model's ranking is substantively correct.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DATASET VALIDATION: FAIL\n{exc}", file=sys.stderr)
        sys.exit(1)
