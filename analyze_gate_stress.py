"""Aggregate controlled-corruption gate calibration results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("acc", "nmi", "ari")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.input_dir.glob("*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for level in sorted({float(row["corruption"]) for row in records}):
        subset = [row for row in records if float(row["corruption"]) == level]
        out = {
            "corruption": level,
            "runs": len(subset),
            "rho_mean": np.mean([row["rho"] for row in subset]),
            "acceptance_rate": np.mean([row["accepted"] for row in subset]),
        }
        for metric in METRICS:
            ungated = np.asarray([
                row["metrics"]["ungated_refinement"][metric]
                - row["metrics"]["observed_graph"][metric]
                for row in subset
            ])
            final = np.asarray([
                row["metrics"]["TRGNMF"][metric]
                - row["metrics"]["observed_graph"][metric]
                for row in subset
            ])
            full = np.asarray([
                row["metrics"]["full_support"][metric]
                - row["metrics"]["observed_graph"][metric]
                for row in subset
            ])
            out[f"ungated_{metric}_gain"] = ungated.mean()
            out[f"final_{metric}_gain"] = final.mean()
            out[f"full_{metric}_gain"] = full.mean()
        rows.append(out)

    decisions = []
    for row in records:
        gain = (
            row["metrics"]["ungated_refinement"]["acc"]
            - row["metrics"]["observed_graph"]["acc"]
        )
        accepted = bool(row["accepted"])
        category = (
            "true_accept" if accepted and gain > 0
            else "false_accept" if accepted and gain < 0
            else "false_reject" if (not accepted) and gain > 0
            else "true_reject" if (not accepted) and gain < 0
            else "neutral"
        )
        decisions.append({
            "dataset": row["dataset"], "seed": row["seed"],
            "corruption": row["corruption"], "rho": row["rho"],
            "accepted": accepted, "ungated_acc_gain": gain,
            "decision_class": category,
        })

    with (args.output_dir / "stress_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "stress_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=decisions[0].keys())
        writer.writeheader(); writer.writerows(decisions)
    counts = {}
    for row in decisions:
        counts[row["decision_class"]] = counts.get(row["decision_class"], 0) + 1
    report = {"runs": len(records), "decision_counts": counts, "levels": rows}
    (args.output_dir / "stress_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
