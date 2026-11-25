"""Matched ablation at the topology selected by the observed graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment_utils import DATASETS, SEEDS, atomic_write, load_dataset
from src.trgnmf import (
    clamped_propagation,
    edge_agreement,
    evaluate_scores,
    gate_coefficient,
    split_labels,
    transition_on_topology,
)


def run_dataset(dataset, data_dir, validated_dir, cache_dir, output_dir):
    x, y, _ = load_dataset(dataset, data_dir)
    validated = {
        int(row["seed"]): row
        for row in json.loads((validated_dir / f"{dataset}.json").read_text(encoding="utf-8"))
    }
    path = output_dir / f"{dataset}.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    complete = {int(row["seed"]) for row in records}
    classes = len(np.unique(y))
    for seed in SEEDS:
        if seed in complete:
            continue
        representation = np.asarray(
            np.load(cache_dir / dataset / f"seed_{seed}.npz")["representation"],
            dtype=np.float32,
        )
        selected = validated[seed]["selected"]
        neighbors = int(selected["neighbors"])
        labelled = split_labels(y, seed, fraction=0.10)
        rho, overlap = edge_agreement(x, representation, neighbors)
        aligned_coefficient, _ = gate_coefficient(rho, threshold=0.5, cap=0.75)
        final_coefficient = aligned_coefficient if selected["accepted"] else 0.0
        permutation = np.random.default_rng(seed + 960_000).permutation(len(x))
        arms = {
            "observed_graph": (0.0, None),
            "TRGNMF": (final_coefficient, None),
            "aligned_candidate": (aligned_coefficient, None),
            "forced_shuffled": (aligned_coefficient, permutation),
            "fixed_half": (0.5, None),
            "uncapped_rho": (max(float(rho), 0.0), None),
            "full_support": (1.0, None),
        }
        metrics = {}
        for name, (coefficient, perm) in arms.items():
            transition = transition_on_topology(
                x,
                representation,
                neighbors,
                coefficient,
                permutation=perm,
            )
            scores = clamped_propagation(transition, y, labelled, classes)
            metrics[name] = evaluate_scores(y, scores, labelled)["unlabelled"]
        records.append(
            {
                "dataset": dataset,
                "seed": seed,
                "neighbors": neighbors,
                "rho": rho,
                "overlap": overlap,
                "aligned_coefficient": aligned_coefficient,
                "final_coefficient": final_coefficient,
                "accepted": bool(selected["accepted"]),
                "metrics": metrics,
            }
        )
        records.sort(key=lambda row: int(row["seed"]))
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
