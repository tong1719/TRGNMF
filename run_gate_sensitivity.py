"""Sensitivity of the revised gate after NMF factor canonicalization."""

from __future__ import annotations

import argparse
import json
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


SETTINGS = (
    *[("threshold", value, value, 0.75) for value in (0.3, 0.4, 0.5, 0.6, 0.7)],
    *[("cap", value, 0.5, value) for value in (0.5, 0.6, 0.75, 0.9, 1.0)],
)


def run_dataset(dataset, data_dir, validated_dir, cache_dir, output_dir):
    x, y, _ = load_dataset(dataset, data_dir)
    validated = {
        int(row["seed"]): row
        for row in json.loads((validated_dir / f"{dataset}.json").read_text(encoding="utf-8"))
    }
    path = output_dir / f"{dataset}.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    complete = {(int(r["seed"]), r["axis"], float(r["value"])) for r in records}
    classes = len(np.unique(y))
    for seed in SEEDS:
        representation = np.asarray(
            np.load(cache_dir / dataset / f"seed_{seed}.npz")["representation"],
            dtype=np.float32,
        )
        labelled = split_labels(y, seed, fraction=0.10)
        neighbors = int(validated[seed]["selected_neighbors"])
        rho, overlap = edge_agreement(x, representation, neighbors)
        observed = transition_on_topology(x, representation, neighbors, 0.0)
        observed_scores = clamped_propagation(observed, y, labelled, classes)
        observed_metrics = evaluate_scores(y, observed_scores, labelled)["unlabelled"]
        for axis, value, threshold, cap in SETTINGS:
            key = (seed, axis, float(value))
            if key in complete:
                continue
            candidate_coefficient, capped = gate_coefficient(rho, threshold, cap)
            candidate = transition_on_topology(
                x, representation, neighbors, candidate_coefficient
            )
            validation = labelled_cross_validation(
                {"observed_graph": observed, "candidate_refinement": candidate},
                y,
                labelled,
                seed + neighbors,
            )
            accepted = bool(
                candidate_coefficient > 0
                and validation["candidate_correct"] > validation["observed_correct"]
                and validation["positive_repeats"] >= 4
            )
            candidate_scores = clamped_propagation(candidate, y, labelled, classes)
            candidate_metrics = evaluate_scores(y, candidate_scores, labelled)["unlabelled"]
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "axis": axis,
                    "value": value,
                    "threshold": threshold,
                    "cap": cap,
                    "neighbors": neighbors,
                    "rho": rho,
                    "overlap": overlap,
                    "candidate_coefficient": candidate_coefficient,
                    "capped": capped,
                    "accepted": accepted,
                    "labelled_cross_validation": validation,
                    "metrics": {
                        "observed_graph": observed_metrics,
                        "candidate_refinement": candidate_metrics,
                        "TRGNMF": candidate_metrics if accepted else observed_metrics,
                    },
                }
            )
            records.sort(key=lambda r: (int(r["seed"]), r["axis"], float(r["value"])))
            atomic_write(path, records)
        print(dataset, seed, "complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        run_dataset(dataset, args.data_dir, args.validated_dir, args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
