"""Calibrate gate decisions under controlled sample-correspondence corruption."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiment_utils import DATASETS, SEEDS, atomic_write, load_dataset
from run_validated_gate import labelled_cross_validation
from src.trgnmf import (
    clamped_propagation,
    edge_agreement,
    evaluate_scores,
    gate_coefficient,
    split_labels,
    transition_on_topology,
)


LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)


def corrupt_representation(z: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction <= 0:
        return z.copy()
    rng = np.random.default_rng(seed)
    count = max(2, int(round(fraction * len(z))))
    chosen = np.sort(rng.choice(len(z), size=count, replace=False))
    shifted = rng.permutation(chosen)
    if np.array_equal(shifted, chosen):
        shifted = np.roll(chosen, 1)
    result = z.copy()
    result[chosen] = z[shifted]
    return result


def run_dataset(
    dataset: str,
    data_dir: Path,
    validated_dir: Path,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    x, y, _ = load_dataset(dataset, data_dir)
    validated = {
        int(row["seed"]): row
        for row in json.loads((validated_dir / f"{dataset}.json").read_text(encoding="utf-8"))
    }
    output = output_dir / f"{dataset}.json"
    records = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    completed = {(int(r["seed"]), float(r["corruption"])) for r in records}
    classes = len(np.unique(y))
    for seed in SEEDS:
        cache = np.load(cache_dir / dataset / f"seed_{seed}.npz")
        representation = np.asarray(cache["representation"], dtype=np.float32)
        labelled = split_labels(y, seed, fraction=0.10)
        neighbors = int(validated[seed]["selected_neighbors"])
        observed = transition_on_topology(x, representation, neighbors, 0.0)
        observed_scores = clamped_propagation(observed, y, labelled, classes)
        observed_metrics = evaluate_scores(y, observed_scores, labelled)["unlabelled"]
        for fraction in LEVELS:
            key = (seed, fraction)
            if key in completed:
                continue
            corrupted = corrupt_representation(
                representation, fraction, seed + int(round(1000 * fraction)) + 950_000
            )
            rho, overlap = edge_agreement(x, corrupted, neighbors)
            candidate_coefficient, capped = gate_coefficient(rho, threshold=0.5, cap=0.75)
            ungated_coefficient = min(max(float(rho), 0.0), 0.75)
            candidate = transition_on_topology(
                x, corrupted, neighbors, candidate_coefficient
            )
            ungated = transition_on_topology(
                x, corrupted, neighbors, ungated_coefficient
            )
            full_support = transition_on_topology(
                x, corrupted, neighbors, 1.0
            )
            validation = labelled_cross_validation(
                {"observed_graph": observed, "candidate_refinement": candidate},
                y,
                labelled,
                seed + neighbors + int(round(1000 * fraction)),
            )
            accepted = bool(
                candidate_coefficient > 0
                and validation["candidate_correct"] > validation["observed_correct"]
                and validation["positive_repeats"] >= 4
            )
            candidate_scores = clamped_propagation(candidate, y, labelled, classes)
            candidate_metrics = evaluate_scores(y, candidate_scores, labelled)["unlabelled"]
            ungated_scores = clamped_propagation(ungated, y, labelled, classes)
            ungated_metrics = evaluate_scores(y, ungated_scores, labelled)["unlabelled"]
            full_scores = clamped_propagation(full_support, y, labelled, classes)
            full_metrics = evaluate_scores(y, full_scores, labelled)["unlabelled"]
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "corruption": fraction,
                    "neighbors": neighbors,
                    "rho": rho,
                    "overlap": overlap,
                    "candidate_coefficient": candidate_coefficient,
                    "ungated_coefficient": ungated_coefficient,
                    "capped": capped,
                    "accepted": accepted,
                    "labelled_cross_validation": validation,
                    "metrics": {
                        "observed_graph": observed_metrics,
                        "candidate_refinement": candidate_metrics,
                        "ungated_refinement": ungated_metrics,
                        "full_support": full_metrics,
                        "TRGNMF": candidate_metrics if accepted else observed_metrics,
                    },
                }
            )
            records.sort(key=lambda r: (int(r["seed"]), float(r["corruption"])))
            atomic_write(output, records)
            print(dataset, seed, fraction, "complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(
            dataset,
            args.data_dir,
            args.validated_dir,
            args.cache_dir,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
