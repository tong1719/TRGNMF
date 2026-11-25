"""Measure the full revised pipeline and tuned observed-graph control."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiment_utils import DATASETS, K_VALUES, atomic_write, load_dataset
from src.trgnmf import (
    canonicalize_nmf_factors,
    clamped_propagation,
    edge_agreement,
    gate_coefficient,
    labelled_cross_validation,
    nmf_factors,
    split_labels,
    transition_on_topology,
)


SEEDS = (57610, 57611, 57612)


def run_dataset(dataset, data_dir, output_dir):
    x, y, _ = load_dataset(dataset, data_dir)
    path = output_dir / f"{dataset}.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    complete = {int(row["seed"]) for row in records}
    classes = len(np.unique(y))
    placeholder = np.zeros((len(x), classes), dtype=np.float32)
    for seed in SEEDS:
        if seed in complete:
            continue
        labelled = split_labels(y, seed, fraction=0.10)

        started = time.perf_counter()
        observed_rows = []
        for neighbors in K_VALUES:
            transition = transition_on_topology(x, placeholder, neighbors, 0.0)
            validation = labelled_cross_validation(
                {"observed_graph": transition}, y, labelled, seed + neighbors
            )
            observed_rows.append((neighbors, transition, validation["observed_graph_accuracy"]))
        observed_choice = max(observed_rows, key=lambda row: (row[2], -row[0]))
        clamped_propagation(observed_choice[1], y, labelled, classes)
        observed_seconds = time.perf_counter() - started

        started = time.perf_counter()
        nmf_started = time.perf_counter()
        raw_w, raw_h, iterations = nmf_factors(x, classes, seed, max_iter=200)
        representation, _ = canonicalize_nmf_factors(raw_w, raw_h)
        representation = representation.astype(np.float32)
        nmf_seconds = time.perf_counter() - nmf_started
        rows = []
        for neighbors in K_VALUES:
            rho, _ = edge_agreement(x, representation, neighbors)
            coefficient, _ = gate_coefficient(rho, threshold=0.5, cap=0.75)
            observed = transition_on_topology(x, representation, neighbors, 0.0)
            candidate = transition_on_topology(x, representation, neighbors, coefficient)
            validation = labelled_cross_validation(
                {"observed_graph": observed, "candidate_refinement": candidate},
                y,
                labelled,
                seed + neighbors,
            )
            rows.append((neighbors, rho, coefficient, observed, candidate, validation))
        chosen = max(
            rows,
            key=lambda row: (row[5]["observed_graph_accuracy"], -row[0]),
        )
        accepted = bool(
            chosen[2] > 0
            and chosen[5]["candidate_refinement_correct"]
            > chosen[5]["observed_graph_correct"]
            and chosen[5]["positive_repeats"] >= 4
        )
        final_transition = chosen[4] if accepted else chosen[3]
        clamped_propagation(final_transition, y, labelled, classes)
        method_seconds = time.perf_counter() - started

        records.append(
            {
                "dataset": dataset,
                "seed": seed,
                "observed_seconds": observed_seconds,
                "TRGNMF_seconds": method_seconds,
                "NMF_seconds": nmf_seconds,
                "overhead_ratio": method_seconds / observed_seconds,
                "selected_neighbors": chosen[0],
                "accepted": accepted,
                "nmf_iterations": iterations,
            }
        )
        records.sort(key=lambda row: int(row["seed"]))
        atomic_write(path, records)
        print(dataset, seed, "complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        run_dataset(dataset, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
