"""Summarize matched ablations for the revised method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = ("acc", "nmi", "ari", "purity", "recall", "fscore")
ARMS = (
    "observed_graph",
    "TRGNMF",
    "aligned_candidate",
    "forced_shuffled",
    "fixed_half",
    "uncapped_rho",
    "full_support",
)


def paired_p(values: np.ndarray, observed: np.ndarray) -> float:
    difference = values - observed
    if np.allclose(difference, 0.0):
        return 1.0
    return float(wilcoxon(difference, zero_method="pratt", alternative="two-sided").pvalue)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    candidate_acc_tests: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        for metric in METRICS:
            observed = np.asarray(
                [record["metrics"]["observed_graph"][metric] for record in records]
            )
            for arm in ARMS:
                values = np.asarray([record["metrics"][arm][metric] for record in records])
                difference = values - observed
                p_value = paired_p(values, observed)
                rows.append(
                    {
                        "dataset": path.stem,
                        "metric": metric,
                        "arm": arm,
                        "runs": len(records),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "gain_over_observed": float(difference.mean()),
                        "paired_wilcoxon_p": p_value,
                    }
                )
                if metric == "acc" and arm == "aligned_candidate":
                    candidate_acc_tests.append(
                        {
                            "dataset": path.stem,
                            "runs": len(records),
                            "mean_gain": float(difference.mean()),
                            "positive": int(np.sum(difference > 0)),
                            "zero": int(np.sum(difference == 0)),
                            "negative": int(np.sum(difference < 0)),
                            "wilcoxon_p": p_value,
                        }
                    )
    with (args.output_dir / "matched_ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row, adjusted in zip(
        candidate_acc_tests,
        holm([float(row["wilcoxon_p"]) for row in candidate_acc_tests]),
    ):
        row["holm_p"] = adjusted
    with (args.output_dir / "candidate_acc_significance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_acc_tests[0]))
        writer.writeheader()
        writer.writerows(candidate_acc_tests)
    print(json.dumps({"datasets": len({row['dataset'] for row in rows}), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
