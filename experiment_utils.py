"""Shared dataset and experiment utilities for TRGNMF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components

from src.trgnmf import (
    _graph_parts,
    canonicalize_nmf_factors,
    clamped_propagation,
    edge_agreement,
    evaluate_scores,
    gate_coefficient,
    nmf_factors,
    partition_metrics,
    row_normalize,
    split_labels,
    transition_on_topology,
)


DATASETS = ("UMIST", "PIE", "COIL20", "COIL100", "MSRA25", "YaleB32")
FILES = {
    "UMIST": "UMIST.mat",
    "PIE": "PIE.mat",
    "COIL20": "COIL20.mat",
    "COIL100": "COIL100.mat",
    "MSRA25": "MSRA25.mat",
    "YaleB32": "YaleB_32x32.mat",
}
EXPECTED = {
    "UMIST": (575, 1024, 20),
    "PIE": (2856, 1024, 68),
    "COIL20": (1440, 1024, 20),
    "COIL100": (7200, 1024, 100),
    "MSRA25": (1799, 256, 12),
    "YaleB32": (2414, 1024, 38),
}
SEEDS = tuple(range(57610, 57630))
K_VALUES = (3, 5, 7, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_dataset(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    path = data_dir / FILES[name]
    data = loadmat(path)
    x_key = "fea" if "fea" in data else "X"
    y_key = "gnd" if "gnd" in data else "Y"
    x = np.asarray(data[x_key], dtype=np.float32)
    _, y = np.unique(np.asarray(data[y_key]).ravel(), return_inverse=True)
    actual = (x.shape[0], x.shape[1], len(np.unique(y)))
    if actual != EXPECTED[name]:
        raise ValueError(f"{name}: expected {EXPECTED[name]}, obtained {actual}")
    if np.min(x) < 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: features must be finite and nonnegative")
    return row_normalize(x).astype(np.float32), y.astype(int), path


def factor_invariance_check(
    sample_factor: np.ndarray,
    component_factor: np.ndarray,
    seed: int,
) -> float:
    canonical, _ = canonicalize_nmf_factors(sample_factor, component_factor)
    rng = np.random.default_rng(seed + 910_000)
    diagonal = np.exp(rng.uniform(-4.0, 4.0, size=sample_factor.shape[1]))
    rescaled_w = sample_factor * diagonal[None, :]
    rescaled_h = component_factor / diagonal[:, None]
    rescaled, _ = canonicalize_nmf_factors(rescaled_w, rescaled_h)
    return float(np.max(np.abs(canonical - rescaled)))


def validation_split(
    y: np.ndarray,
    labelled: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 920_000)
    train: list[int] = []
    validation: list[int] = []
    labelled = np.asarray(labelled, dtype=int)
    for class_id in np.unique(y):
        pool = labelled[y[labelled] == class_id]
        count = min(max(1, int(round(0.2 * len(pool)))), len(pool) - 1)
        chosen = np.sort(rng.choice(pool, size=count, replace=False))
        validation.extend(chosen.tolist())
        train.extend(np.setdiff1d(pool, chosen, assume_unique=True).tolist())
    return np.asarray(sorted(train), dtype=int), np.asarray(sorted(validation), dtype=int)


def component_diagnostics(
    features: np.ndarray,
    labelled: np.ndarray,
    neighbors: int,
) -> dict[str, int | float]:
    _, ids, _ = _graph_parts(features, neighbors)
    rows = np.repeat(np.arange(len(features)), neighbors)
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, ids.ravel())),
        shape=(len(features), len(features)),
    )
    adjacency = adjacency.maximum(adjacency.T)
    count, membership = connected_components(adjacency, directed=False)
    labelled_components = set(membership[np.asarray(labelled, dtype=int)].tolist())
    unlabelled_components = [component for component in range(count) if component not in labelled_components]
    affected = int(np.sum(np.isin(membership, unlabelled_components)))
    return {
        "components": int(count),
        "components_without_labels": int(len(unlabelled_components)),
        "samples_in_components_without_labels": affected,
        "sample_fraction_without_label_reach": float(affected / len(features)),
    }


def validation_accuracy(
    transition: sparse.spmatrix,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    classes: int,
) -> float:
    scores = clamped_propagation(transition, y, train, classes)
    return float(np.mean(np.argmax(scores[validation], axis=1) == y[validation]))


def run_seed(name: str, x: np.ndarray, y: np.ndarray, source: Path, seed: int) -> dict[str, object]:
    classes = len(np.unique(y))
    labelled = split_labels(y, seed, fraction=0.10)
    train, validation = validation_split(y, labelled, seed)
    raw_w, raw_h, iterations = nmf_factors(x, classes, seed, max_iter=200)
    representation, _ = canonicalize_nmf_factors(raw_w, raw_h)
    representation = representation.astype(np.float32)
    invariance_error = factor_invariance_check(raw_w, raw_h, seed)
    permutation = np.random.default_rng(seed + 930_000).permutation(len(x))

    settings: list[dict[str, object]] = []
    for neighbors in K_VALUES:
        rho, overlap = edge_agreement(x, representation, neighbors)
        coefficient, capped = gate_coefficient(rho, threshold=0.5, cap=0.75)
        transitions = {
            "observed_graph": transition_on_topology(x, representation, neighbors, 0.0),
            "TRGNMF": transition_on_topology(x, representation, neighbors, coefficient),
            "full_support": transition_on_topology(x, representation, neighbors, 1.0),
            "forced_shuffled": transition_on_topology(
                x, representation, neighbors, coefficient, permutation=permutation
            ),
        }
        validation_scores = {
            arm: validation_accuracy(transition, y, train, validation, classes)
            for arm, transition in transitions.items()
        }
        final_metrics = {}
        for arm, transition in transitions.items():
            scores = clamped_propagation(transition, y, labelled, classes)
            final_metrics[arm] = evaluate_scores(y, scores, labelled)["unlabelled"]
        settings.append(
            {
                "neighbors": int(neighbors),
                "rho": float(rho),
                "overlap": float(overlap),
                "coefficient": float(coefficient),
                "capped": bool(capped),
                "validation_accuracy": validation_scores,
                "metrics": final_metrics,
                "components": component_diagnostics(x, labelled, neighbors),
            }
        )

    selected = {}
    for arm in ("observed_graph", "TRGNMF"):
        best = max(
            settings,
            key=lambda row: (float(row["validation_accuracy"][arm]), -int(row["neighbors"])),
        )
        selected[arm] = {
            "neighbors": int(best["neighbors"]),
            "validation_accuracy": float(best["validation_accuracy"][arm]),
            "metrics": best["metrics"][arm],
        }

    return {
        "dataset": name,
        "seed": int(seed),
        "shape": list(x.shape),
        "classes": int(classes),
        "labelled": int(len(labelled)),
        "tuning_train_labels": int(len(train)),
        "tuning_validation_labels": int(len(validation)),
        "data_sha256": sha256(source),
        "nmf_iterations": int(iterations),
        "factor_canonicalization": "component_l2",
        "scale_invariance_max_abs_error": invariance_error,
        "settings": settings,
        "label_only_selected": selected,
    }


def run_dataset(name: str, data_dir: Path, output_dir: Path, limit: int | None) -> None:
    x, y, source = load_dataset(name, data_dir)
    output = output_dir / f"{name}.json"
    records = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    completed = {int(record["seed"]) for record in records}
    seeds = SEEDS if limit is None else SEEDS[:limit]
    for seed in seeds:
        if seed in completed:
            print(name, seed, "already complete", flush=True)
            continue
        records.append(run_seed(name, x, y, source, seed))
        records.sort(key=lambda record: int(record["seed"]))
        atomic_write(output, records)
        print(name, seed, "complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/central"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run_dataset(dataset, args.data_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
