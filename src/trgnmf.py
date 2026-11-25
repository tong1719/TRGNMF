"""Core implementation of the revised TRGNMF method.

The implementation follows the submitted equations and preserves the numerical
choices of the experiments: row-normalized nonnegative features,
rank-C NMF, scale-canonicalized factors, self-tuning k-NN affinities, a
two-stage reliability gate, bounded refinement on the observed topology, and
clamped label propagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


EPS = 1.0e-12


@dataclass(frozen=True)
class GraphDiagnostics:
    rho: float
    overlap: float
    coefficient: float
    capped: bool


def row_normalize(x: np.ndarray) -> np.ndarray:
    """Project to the nonnegative orthant and normalize every sample row."""
    values = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1.0e-10)


def split_labels(y: np.ndarray, seed: int, fraction: float = 0.10) -> np.ndarray:
    """Draw the fixed labelled fraction independently within every class."""
    labels = np.asarray(y, dtype=int).ravel()
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in np.unique(labels):
        pool = np.flatnonzero(labels == class_id)
        count = max(1, int(round(fraction * len(pool))))
        selected.extend(rng.choice(pool, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def canonicalize_nmf_factors(
    sample_factor: np.ndarray,
    component_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the positive diagonal scaling ambiguity of an NMF factor pair.

    For ``X ~= W H``, the pair ``W D`` and ``D^{-1} H`` gives the same
    reconstruction for any positive diagonal ``D``.  Normalizing rows of ``W``
    alone does not remove that ambiguity.  We therefore normalize every row of
    ``H`` to unit L2 norm and absorb its norm into the corresponding column of
    ``W`` before the sample-wise row normalization used by TRGNMF.
    """
    w = np.maximum(np.asarray(sample_factor, dtype=np.float64), 0.0)
    h = np.maximum(np.asarray(component_factor, dtype=np.float64), 0.0)
    scale = np.maximum(np.linalg.norm(h, axis=1), EPS)
    canonical_w = w * scale[None, :]
    canonical_h = h / scale[:, None]
    return row_normalize(canonical_w), canonical_h


def nmf_factors(
    x: np.ndarray,
    rank: int,
    seed: int,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the raw rank-C factor pair and the realized iteration count."""
    model = NMF(
        n_components=rank,
        init="random",
        random_state=seed,
        solver="cd",
        beta_loss="frobenius",
        max_iter=max_iter,
        tol=1.0e-4,
        shuffle=False,
    )
    sample_factor = model.fit_transform(np.asarray(x, dtype=np.float32))
    return (
        np.asarray(sample_factor, dtype=np.float64),
        np.asarray(model.components_, dtype=np.float64),
        int(model.n_iter_),
    )


def nmf_embedding(
    x: np.ndarray,
    rank: int,
    seed: int,
    max_iter: int = 200,
) -> tuple[np.ndarray, int]:
    """Return the scale-canonicalized rank-C NMF sample representation."""
    sample_factor, component_factor, iterations = nmf_factors(
        x, rank, seed, max_iter=max_iter
    )
    representation, _ = canonicalize_nmf_factors(sample_factor, component_factor)
    return representation.astype(np.float32), iterations


def _graph_parts(x: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance, indices = NearestNeighbors(n_neighbors=neighbors + 1).fit(x).kneighbors(x)
    distance = distance[:, 1:]
    indices = indices[:, 1:]
    scale = np.maximum(distance[:, -1], EPS)
    return distance, indices, scale


def edge_agreement(
    x: np.ndarray,
    representation: np.ndarray,
    neighbors: int,
) -> tuple[float, float]:
    """Spearman agreement and neighbor overlap on observed directed edges."""
    raw_distance, raw_ids, raw_scale = _graph_parts(x, neighbors)
    _, latent_ids, latent_scale = _graph_parts(representation, neighbors)
    rows = np.repeat(np.arange(len(x)), neighbors)
    cols = raw_ids.ravel()
    observed = np.exp(
        -(raw_distance.ravel() ** 2)
        / np.maximum(raw_scale[rows] * raw_scale[cols], EPS)
    )
    latent_distance = np.sum((representation[rows] - representation[cols]) ** 2, axis=1)
    support = np.exp(
        -latent_distance / np.maximum(latent_scale[rows] * latent_scale[cols], EPS)
    )
    statistic = spearmanr(observed, support).statistic
    rho = float(statistic) if np.isfinite(statistic) else -1.0
    overlap = float(
        np.mean(
            [
                len(set(raw_ids[i]).intersection(latent_ids[i])) / neighbors
                for i in range(len(x))
            ]
        )
    )
    return rho, overlap


def gate_coefficient(rho: float, threshold: float = 0.5, cap: float = 0.75) -> tuple[float, bool]:
    """Apply the reliability threshold and upper bound from the manuscript."""
    uncapped = float(np.clip(rho, 0.0, 1.0)) if rho >= threshold else 0.0
    return min(uncapped, cap), bool(uncapped > cap)


def transition_on_topology(
    x: np.ndarray,
    representation: np.ndarray,
    neighbors: int,
    coefficient: float,
    permutation: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Construct the row-stochastic refined graph without adding neighbors."""
    raw_distance, raw_ids, raw_scale = _graph_parts(x, neighbors)
    rows = np.repeat(np.arange(len(x)), neighbors)
    cols = raw_ids.ravel()
    observed = np.exp(
        -(raw_distance.ravel() ** 2)
        / np.maximum(raw_scale[rows] * raw_scale[cols], EPS)
    )

    if coefficient <= 0.0:
        values = observed
    else:
        support_space = representation if permutation is None else representation[permutation]
        _, _, latent_scale = _graph_parts(support_space, neighbors)
        latent_distance = np.sum((support_space[rows] - support_space[cols]) ** 2, axis=1)
        support = np.exp(
            -latent_distance / np.maximum(latent_scale[rows] * latent_scale[cols], EPS)
        )
        values = observed * ((1.0 - coefficient) + coefficient * support)
    adjacency = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(x), len(x)), dtype=np.float64
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(1.0 / np.maximum(degree, EPS)) @ adjacency).tocsr()


def clamped_propagation(
    transition: sparse.spmatrix,
    y: np.ndarray,
    labelled: np.ndarray,
    classes: int,
    iterations: int = 50,
) -> np.ndarray:
    """Run the clamped propagation used in all frozen experiments."""
    labels = np.asarray(y, dtype=int).ravel()
    labelled = np.asarray(labelled, dtype=int)
    target = np.zeros((len(labels), classes), dtype=np.float64)
    target[labelled, labels[labelled]] = 1.0
    scores = target.copy()
    prior = np.bincount(labels[labelled], minlength=classes).astype(float)
    prior /= prior.sum()
    unlabelled = np.ones(len(labels), dtype=bool)
    unlabelled[labelled] = False
    scores[unlabelled] = prior
    for _ in range(iterations):
        # The common 0.9 coefficient cancels on unlabelled rows during the
        # subsequent row normalization; labelled rows are clamped exactly.
        scores = 0.9 * (transition @ scores) + 0.1 * target
        scores[labelled] = target[labelled]
        scores /= np.maximum(scores.sum(axis=1, keepdims=True), 1.0e-10)
    return scores


def labelled_cross_validation(
    transitions: dict[str, sparse.spmatrix],
    y: np.ndarray,
    labelled: np.ndarray,
    seed: int,
    repeats: int = 5,
) -> dict[str, float | int]:
    """Compare transitions using repeated stratified folds of known labels only."""
    labels = np.asarray(y, dtype=int).ravel()
    labelled = np.asarray(labelled, dtype=int)
    class_counts = np.bincount(labels[labelled])
    positive = class_counts[class_counts > 0]
    folds = int(min(5, positive.min()))
    if folds < 2:
        raise ValueError("At least two labelled samples per class are required")
    correct = {name: 0 for name in transitions}
    total = 0
    local_y = labels[labelled]
    classes = len(np.unique(labels))
    repeat_correct_rows: list[dict[str, int]] = []
    for repeat in range(repeats):
        repeat_correct = {name: 0 for name in transitions}
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + 940_000 + 1009 * repeat,
        )
        for train_pos, validation_pos in splitter.split(labelled, local_y):
            train = labelled[train_pos]
            validation = labelled[validation_pos]
            total += len(validation)
            for name, transition in transitions.items():
                scores = clamped_propagation(transition, labels, train, classes)
                value = int(np.sum(np.argmax(scores[validation], axis=1) == labels[validation]))
                correct[name] += value
                repeat_correct[name] += value
        repeat_correct_rows.append(repeat_correct)
    result: dict[str, float | int] = {
        "folds": folds,
        "repeats": repeats,
        "total": total,
    }
    for name in transitions:
        result[f"{name}_correct"] = correct[name]
        result[f"{name}_accuracy"] = correct[name] / total
    if {"observed_graph", "candidate_refinement"}.issubset(transitions):
        differences = [
            (row["candidate_refinement"] - row["observed_graph"]) / len(labelled)
            for row in repeat_correct_rows
        ]
        result["repeat_accuracy_differences"] = differences
        result["positive_repeats"] = int(np.sum(np.asarray(differences) > 0))
    return result


def _choose_two(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return values * (values - 1) // 2


def partition_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the six clustering measures reported in the manuscript."""
    truth = np.asarray(y_true, dtype=int).ravel()
    predicted = np.asarray(y_pred, dtype=int).ravel()
    _, true_index = np.unique(truth, return_inverse=True)
    _, pred_index = np.unique(predicted, return_inverse=True)
    table = np.zeros(
        (int(true_index.max()) + 1, int(pred_index.max()) + 1), dtype=np.int64
    )
    np.add.at(table, (true_index, pred_index), 1)
    true_pairs = int(np.sum(_choose_two(np.sum(table, axis=1))))
    predicted_pairs = int(np.sum(_choose_two(np.sum(table, axis=0))))
    true_positive = int(np.sum(_choose_two(table)))
    precision = true_positive / predicted_pairs if predicted_pairs else 1.0
    recall = true_positive / true_pairs if true_pairs else 1.0
    fscore = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    purity = float(np.sum(np.max(table, axis=0)) / len(truth))
    return {
        "acc": float(np.mean(truth == predicted)),
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ari": float(adjusted_rand_score(truth, predicted)),
        "purity": purity,
        "recall": float(recall),
        "fscore": float(fscore),
        "pairwise_precision_diagnostic": float(precision),
    }


def evaluate_scores(
    y: np.ndarray,
    scores: np.ndarray,
    labelled: np.ndarray,
) -> dict[str, dict[str, float]]:
    predicted = np.argmax(scores, axis=1)
    unlabelled = np.ones(len(y), dtype=bool)
    unlabelled[np.asarray(labelled, dtype=int)] = False
    return {
        "all": partition_metrics(y, predicted),
        "unlabelled": partition_metrics(np.asarray(y)[unlabelled], predicted[unlabelled]),
    }


def run_pair(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    label_fraction: float = 0.10,
    neighbor_grid: tuple[int, ...] = (3, 5, 7, 9),
    threshold: float = 0.5,
    cap: float = 0.75,
    nmf_iterations: int = 200,
    propagation_iterations: int = 50,
) -> dict[str, object]:
    """Run revised TRGNMF and its matched observed-graph control.

    The observed graph alone selects ``k`` by repeated cross-validation on the
    available labelled samples.  At each candidate ``k``, the structural
    Spearman screen proposes a bounded coefficient.  The refinement is used
    only when its aggregate labelled-sample validation score is strictly
    higher than that of the observed graph and it improves in at least four of
    the five repeated validation partitions.  Unlabelled evaluation labels are
    not used by either decision.
    """
    features = row_normalize(x).astype(np.float32)
    labels = np.asarray(y, dtype=int).ravel()
    _, labels = np.unique(labels, return_inverse=True)
    classes = len(np.unique(labels))
    labelled = split_labels(labels, seed, label_fraction)
    representation, actual_iterations = nmf_embedding(
        features, classes, seed, max_iter=nmf_iterations
    )
    settings: list[dict[str, object]] = []
    for neighbors in neighbor_grid:
        rho, overlap = edge_agreement(features, representation, neighbors)
        candidate_coefficient, capped = gate_coefficient(rho, threshold, cap)
        observed = transition_on_topology(features, representation, neighbors, 0.0)
        candidate = transition_on_topology(
            features, representation, neighbors, candidate_coefficient
        )
        validation = labelled_cross_validation(
            {
                "observed_graph": observed,
                "candidate_refinement": candidate,
            },
            labels,
            labelled,
            seed + neighbors,
            repeats=5,
        )
        accepted = bool(
            candidate_coefficient > 0.0
            and validation["candidate_refinement_correct"]
            > validation["observed_graph_correct"]
            and validation["positive_repeats"] >= 4
        )
        final_coefficient = candidate_coefficient if accepted else 0.0
        final_transition = candidate if accepted else observed
        settings.append(
            {
                "neighbors": int(neighbors),
                "rho": float(rho),
                "overlap": float(overlap),
                "candidate_coefficient": float(candidate_coefficient),
                "accepted": accepted,
                "final_coefficient": float(final_coefficient),
                "capped": bool(capped),
                "labelled_cross_validation": validation,
                "transitions": {
                    "observed_graph": observed,
                    "TRGNMF": final_transition,
                    "candidate_refinement": candidate,
                },
            }
        )
    selected = max(
        settings,
        key=lambda row: (
            float(row["labelled_cross_validation"]["observed_graph_accuracy"]),
            -int(row["neighbors"]),
        ),
    )
    arms: dict[str, object] = {}
    for name, transition in selected["transitions"].items():
        scores = clamped_propagation(
            transition, labels, labelled, classes, propagation_iterations
        )
        coefficient = (
            0.0
            if name == "observed_graph"
            else selected["candidate_coefficient"]
            if name == "candidate_refinement"
            else selected["final_coefficient"]
        )
        arms[name] = {
            "coefficient": float(coefficient),
            "metrics": evaluate_scores(labels, scores, labelled),
        }
    for row in settings:
        row.pop("transitions")
    return {
        "seed": int(seed),
        "shape": list(features.shape),
        "classes": int(classes),
        "labelled": int(len(labelled)),
        "minimum_labels_per_class": int(
            min(np.sum(labels[labelled] == class_id) for class_id in range(classes))
        ),
        "neighbor_grid": [int(value) for value in neighbor_grid],
        "selection_rule": "observed-graph labelled cross-validation; smallest k breaks ties",
        "selected_neighbors": int(selected["neighbors"]),
        "rho": float(selected["rho"]),
        "overlap": float(selected["overlap"]),
        "candidate_coefficient": float(selected["candidate_coefficient"]),
        "accepted": bool(selected["accepted"]),
        "accepted_coefficient": float(selected["final_coefficient"]),
        "capped": bool(selected["capped"]),
        "labelled_cross_validation": selected["labelled_cross_validation"],
        "nmf_iterations": actual_iterations,
        "factor_canonicalization": "component_l2",
        "arms": arms,
        "settings": settings,
    }


def clustering_accuracy_with_assignment(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optional permutation-invariant clustering accuracy for external use."""
    truth = np.asarray(y_true, dtype=int).ravel()
    predicted = np.asarray(y_pred, dtype=int).ravel()
    size = max(int(truth.max()), int(predicted.max())) + 1
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (predicted, truth), 1)
    row, col = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[row, col].sum() / len(truth))
