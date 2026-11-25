"""Evaluate the revised two-stage reliability gate under a matched protocol.

Stage 1 retains the label-free rank-agreement screen.  Stage 2 uses only the
available labelled subset in stratified cross-validation to check whether the
candidate refinement improves propagation relative to the observed graph.
Unlabelled evaluation labels are never used for either k selection or gating.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiment_utils import (
    DATASETS,
    K_VALUES,
    SEEDS,
    atomic_write,
    component_diagnostics,
    load_dataset,
)
from src.trgnmf import (
    canonicalize_nmf_factors,
    clamped_propagation,
    edge_agreement,
    evaluate_scores,
    gate_coefficient,
    nmf_factors,
    split_labels,
    transition_on_topology,
)


def labelled_cross_validation(
    transitions: dict[str, object],
    y: np.ndarray,
    labelled: np.ndarray,
    seed: int,
    repeats: int = 5,
) -> dict[str, float | int]:
    """Return repeated out-of-fold accuracy for observed and refined graphs."""
    labelled = np.asarray(labelled, dtype=int)
    class_counts = np.bincount(y[labelled])
    positive = class_counts[class_counts > 0]
    folds = int(min(5, positive.min()))
    if folds < 2:
        raise ValueError("At least two labelled samples per class are required")
    correct = {name: 0 for name in transitions}
    total = 0
    local_y = y[labelled]
    repeat_differences: list[float] = []
    for repeat in range(repeats):
        repeat_correct = {name: 0 for name in transitions}
        repeat_total = 0
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + 940_000 + 1009 * repeat,
        )
        for train_pos, validation_pos in splitter.split(labelled, local_y):
            train = labelled[train_pos]
            validation = labelled[validation_pos]
            total += len(validation)
            repeat_total += len(validation)
            for name, transition in transitions.items():
                scores = clamped_propagation(transition, y, train, len(np.unique(y)))
                value = int(np.sum(np.argmax(scores[validation], axis=1) == y[validation]))
                correct[name] += value
                repeat_correct[name] += value
        repeat_differences.append(
            (repeat_correct["candidate_refinement"] - repeat_correct["observed_graph"])
            / repeat_total
        )
    return {
        "folds": folds,
        "repeats": repeats,
        "observed_correct": correct["observed_graph"],
        "candidate_correct": correct["candidate_refinement"],
        "total": total,
        "observed_accuracy": correct["observed_graph"] / total,
        "candidate_accuracy": correct["candidate_refinement"] / total,
        "accuracy_difference": (correct["candidate_refinement"] - correct["observed_graph"]) / total,
        "repeat_accuracy_differences": repeat_differences,
        "positive_repeats": int(np.sum(np.asarray(repeat_differences) > 0)),
    }


def embedding_cache_path(cache_dir: Path, dataset: str, seed: int) -> Path:
    return cache_dir / dataset / f"seed_{seed}.npz"


def obtain_embedding(
    x: np.ndarray,
    classes: int,
    dataset: str,
    seed: int,
    cache_dir: Path,
) -> tuple[np.ndarray, int]:
    path = embedding_cache_path(cache_dir, dataset, seed)
    if path.exists():
        cached = np.load(path)
        return np.asarray(cached["representation"], dtype=np.float32), int(cached["iterations"])
    raw_w, raw_h, iterations = nmf_factors(x, classes, seed, max_iter=200)
    representation, _ = canonicalize_nmf_factors(raw_w, raw_h)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, representation=representation.astype(np.float32), iterations=iterations)
    os.replace(temporary, path)
    return representation.astype(np.float32), iterations


def run_seed(
    dataset: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    cache_dir: Path,
) -> dict[str, object]:
    classes = len(np.unique(y))
    labelled = split_labels(y, seed, fraction=0.10)
    representation, iterations = obtain_embedding(x, classes, dataset, seed, cache_dir)
    settings: list[dict[str, object]] = []
    for neighbors in K_VALUES:
        rho, overlap = edge_agreement(x, representation, neighbors)
        candidate_coefficient, capped = gate_coefficient(rho, threshold=0.5, cap=0.75)
        observed = transition_on_topology(x, representation, neighbors, 0.0)
        candidate = transition_on_topology(x, representation, neighbors, candidate_coefficient)
        validation = labelled_cross_validation(
            {"observed_graph": observed, "candidate_refinement": candidate},
            y,
            labelled,
            seed + neighbors,
        )
        # A strict positive paired validation gain is required.  Ties fall back
        # to the observed graph, making the safeguard conservative by design.
        accepted = bool(
            candidate_coefficient > 0.0
            and validation["candidate_correct"] > validation["observed_correct"]
            and validation["positive_repeats"] >= 4
        )
        coefficient = candidate_coefficient if accepted else 0.0
        final_transition = candidate if accepted else observed
        observed_scores = clamped_propagation(observed, y, labelled, classes)
        final_scores = clamped_propagation(final_transition, y, labelled, classes)
        settings.append(
            {
                "neighbors": neighbors,
                "rho": rho,
                "overlap": overlap,
                "candidate_coefficient": candidate_coefficient,
                "accepted": accepted,
                "final_coefficient": coefficient,
                "capped": capped,
                "labelled_cross_validation": validation,
                "metrics": {
                    "observed_graph": evaluate_scores(y, observed_scores, labelled)["unlabelled"],
                    "TRGNMF": evaluate_scores(y, final_scores, labelled)["unlabelled"],
                    "candidate_refinement": evaluate_scores(
                        y,
                        clamped_propagation(candidate, y, labelled, classes),
                        labelled,
                    )["unlabelled"],
                },
                "components": component_diagnostics(x, labelled, neighbors),
            }
        )
    # The observed-graph cross-validation score selects k for both arms; this
    # prevents the proposed method from receiving a more favorable topology.
    selected = max(
        settings,
        key=lambda row: (
            float(row["labelled_cross_validation"]["observed_accuracy"]),
            -int(row["neighbors"]),
        ),
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "classes": classes,
        "labelled": len(labelled),
        "nmf_iterations": iterations,
        "selection_rule": "observed-graph labelled cross-validation; smallest k breaks ties",
        "selected_neighbors": selected["neighbors"],
        "selected": selected,
        "settings": settings,
    }


def run_dataset(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    limit: int | None,
) -> None:
    x, y, _ = load_dataset(dataset, data_dir)
    path = output_dir / f"{dataset}.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    complete = {int(record["seed"]) for record in records}
    seeds = SEEDS if limit is None else SEEDS[:limit]
    for seed in seeds:
        if seed in complete:
            print(dataset, seed, "already complete", flush=True)
            continue
        records.append(run_seed(dataset, x, y, seed, cache_dir))
        records.sort(key=lambda record: int(record["seed"]))
        atomic_write(path, records)
        print(dataset, seed, "complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/validated_gate"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/embedding_cache"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(dataset, args.data_dir, args.output_dir, args.cache_dir, args.limit)


if __name__ == "__main__":
    main()
