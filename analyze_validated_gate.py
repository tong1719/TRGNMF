"""Summarize the revised two-stage gate without using evaluation labels for tuning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon


METRICS = ("acc", "nmi", "ari", "purity", "recall", "fscore")


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

    datasets: dict[str, list[dict]] = {}
    for path in sorted(args.input_dir.glob("*.json")):
        datasets[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    p_values = []
    gate_rows = []
    selection = {}
    components = []
    for dataset, records in datasets.items():
        selection[dataset] = {}
        for record in records:
            selected = record["selected"]
            k = str(selected["neighbors"])
            selection[dataset][k] = selection[dataset].get(k, 0) + 1
            candidate_gain = (
                selected["metrics"]["candidate_refinement"]["acc"]
                - selected["metrics"]["observed_graph"]["acc"]
            )
            accepted = bool(selected["accepted"])
            gate_rows.append(
                {
                    "dataset": dataset,
                    "seed": record["seed"],
                    "k": int(k),
                    "rho": selected["rho"],
                    "cv_difference": selected["labelled_cross_validation"]["accuracy_difference"],
                    "positive_repeats": selected["labelled_cross_validation"]["positive_repeats"],
                    "accepted": accepted,
                    "candidate_acc_gain": candidate_gain,
                    "decision_class": (
                        "true_accept" if accepted and candidate_gain > 0
                        else "false_accept" if accepted and candidate_gain < 0
                        else "false_reject" if (not accepted) and candidate_gain > 0
                        else "true_reject" if (not accepted) and candidate_gain < 0
                        else "neutral"
                    ),
                }
            )
            comp = selected["components"].copy()
            comp.update({"dataset": dataset, "seed": record["seed"], "k": int(k)})
            components.append(comp)

        for metric in METRICS:
            observed = np.asarray(
                [r["selected"]["metrics"]["observed_graph"][metric] for r in records]
            )
            proposed = np.asarray(
                [r["selected"]["metrics"]["TRGNMF"][metric] for r in records]
            )
            difference = proposed - observed
            try:
                p_value = float(wilcoxon(difference, zero_method="pratt").pvalue)
            except ValueError:
                p_value = 1.0
            p_values.append(p_value)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "observed_mean": observed.mean(),
                    "observed_std": observed.std(ddof=1),
                    "TRGNMF_mean": proposed.mean(),
                    "TRGNMF_std": proposed.std(ddof=1),
                    "mean_gain": difference.mean(),
                    "positive": int(np.sum(difference > 0)),
                    "zero": int(np.sum(difference == 0)),
                    "negative": int(np.sum(difference < 0)),
                    "wilcoxon_p": p_value,
                }
            )
    for row, adjusted in zip(rows, holm(p_values)):
        row["holm_p"] = adjusted

    with (args.output_dir / "primary_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "gate_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=gate_rows[0].keys())
        writer.writeheader(); writer.writerows(gate_rows)
    with (args.output_dir / "components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=components[0].keys())
        writer.writeheader(); writer.writerows(components)

    rho_rows = []
    for dataset in sorted(datasets):
        values = np.asarray(
            [row["rho"] for row in gate_rows if row["dataset"] == dataset],
            dtype=float,
        )
        rho_rows.append(
            {
                "dataset": dataset,
                "runs": len(values),
                "rho_mean": float(values.mean()),
                "rho_std": float(values.std(ddof=1)),
                "rho_below_cap_runs": int(np.sum(values < 0.75)),
                "rho_below_cap_fraction": float(np.mean(values < 0.75)),
                "rho_min": float(values.min()),
                "rho_max": float(values.max()),
            }
        )
    with (args.output_dir / "rho_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rho_rows[0].keys())
        writer.writeheader(); writer.writerows(rho_rows)

    decision_counts = {}
    for row in gate_rows:
        decision_counts[row["decision_class"]] = decision_counts.get(row["decision_class"], 0) + 1
    accepted_rows = [row for row in gate_rows if row["accepted"]]
    rejected_rows = [row for row in gate_rows if not row["accepted"]]
    rho_gain = spearmanr(
        [row["rho"] for row in gate_rows],
        [row["candidate_acc_gain"] for row in gate_rows],
    )
    rho_gain_rows = []
    for dataset in sorted(datasets):
        subset = [row for row in gate_rows if row["dataset"] == dataset]
        association = spearmanr(
            [row["rho"] for row in subset],
            [row["candidate_acc_gain"] for row in subset],
        )
        rho_gain_rows.append(
            {
                "scope": dataset,
                "runs": len(subset),
                "spearman": float(association.statistic),
                "pvalue": float(association.pvalue),
            }
        )
    rho_rank = rankdata([row["rho"] for row in gate_rows])
    gain_rank = rankdata([row["candidate_acc_gain"] for row in gate_rows])
    dataset_names = sorted(datasets)
    dataset_design = np.column_stack(
        [
            np.asarray(
                [row["dataset"] == dataset for row in gate_rows], dtype=float
            )
            for dataset in dataset_names
        ]
    )
    rho_residual = rho_rank - dataset_design @ np.linalg.lstsq(
        dataset_design, rho_rank, rcond=None
    )[0]
    gain_residual = gain_rank - dataset_design @ np.linalg.lstsq(
        dataset_design, gain_rank, rcond=None
    )[0]
    dataset_adjusted_rank_correlation = float(
        np.corrcoef(rho_residual, gain_residual)[0, 1]
    )
    rho_gain_rows.extend(
        [
            {
                "scope": "pooled_descriptive",
                "runs": len(gate_rows),
                "spearman": float(rho_gain.statistic),
                "pvalue": float(rho_gain.pvalue),
            },
            {
                "scope": "dataset_adjusted_rank_residuals",
                "runs": len(gate_rows),
                "spearman": dataset_adjusted_rank_correlation,
                "pvalue": "",
            },
        ]
    )
    with (args.output_dir / "rho_gain_association.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rho_gain_rows[0].keys())
        writer.writeheader(); writer.writerows(rho_gain_rows)
    repeat_sensitivity = {}
    for minimum_positive_repeats in (3, 4, 5):
        decisions = [
            bool(
                row["rho"] >= 0.5
                and row["cv_difference"] > 0
                and row["positive_repeats"] >= minimum_positive_repeats
            )
            for row in gate_rows
        ]
        repeat_sensitivity[str(minimum_positive_repeats)] = {
            "accepted_runs": int(sum(decisions)),
            "accepted_beneficial_runs": int(
                sum(
                    decision and row["candidate_acc_gain"] > 0
                    for decision, row in zip(decisions, gate_rows)
                )
            ),
            "accepted_neutral_runs": int(
                sum(
                    decision and row["candidate_acc_gain"] == 0
                    for decision, row in zip(decisions, gate_rows)
                )
            ),
            "accepted_harmful_runs": int(
                sum(
                    decision and row["candidate_acc_gain"] < 0
                    for decision, row in zip(decisions, gate_rows)
                )
            ),
            "mean_final_acc_gain": float(
                np.mean(
                    [
                        row["candidate_acc_gain"] if decision else 0.0
                        for decision, row in zip(decisions, gate_rows)
                    ]
                )
            ),
        }
    rho_values = np.asarray([row["rho"] for row in gate_rows], dtype=float)
    report = {
        "datasets": len(datasets),
        "runs": len(gate_rows),
        "selection_frequencies": selection,
        "gate_decision_counts": decision_counts,
        "accepted_runs": len(accepted_rows),
        "accepted_beneficial_runs": int(
            sum(row["candidate_acc_gain"] > 0 for row in accepted_rows)
        ),
        "accepted_neutral_runs": int(
            sum(row["candidate_acc_gain"] == 0 for row in accepted_rows)
        ),
        "accepted_harmful_runs": int(
            sum(row["candidate_acc_gain"] < 0 for row in accepted_rows)
        ),
        "rejected_runs": len(rejected_rows),
        "rejected_neutral_runs": int(
            sum(row["candidate_acc_gain"] == 0 for row in rejected_rows)
        ),
        "acceptance_rate": float(np.mean([row["accepted"] for row in gate_rows])),
        "primary_structural_rejections": int(
            sum(float(row["rho"]) < 0.5 for row in gate_rows)
        ),
        "rho_candidate_gain_pooled_spearman_descriptive": float(rho_gain.statistic),
        "rho_candidate_gain_pooled_spearman_p": float(rho_gain.pvalue),
        "rho_candidate_gain_dataset_adjusted_rank_correlation": (
            dataset_adjusted_rank_correlation
        ),
        "rho_below_cap_runs": int(np.sum(rho_values < 0.75)),
        "rho_below_cap_fraction": float(np.mean(rho_values < 0.75)),
        "label_positive_repeat_sensitivity": repeat_sensitivity,
        "mean_candidate_acc_gain": float(np.mean([row["candidate_acc_gain"] for row in gate_rows])),
        "mean_final_acc_gain": float(
            np.mean(
                [
                    row["candidate_acc_gain"] if row["accepted"] else 0.0
                    for row in gate_rows
                ]
            )
        ),
        "no_label_component_runs": int(
            sum(row["components_without_labels"] > 0 for row in components)
        ),
        "maximum_sample_fraction_without_label_reach": float(
            max(row["sample_fraction_without_label_reach"] for row in components)
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
