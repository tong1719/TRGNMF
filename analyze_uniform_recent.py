"""Summarize the additional-dataset experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = ("acc", "nmi", "ari", "purity", "recall", "fscore")
ARMS = (
    "observed_graph",
    "TRGNMF",
    "candidate_refinement",
    "ungated_refinement",
    "full_support",
    "forced_shuffled",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        for setting_name in ("uniform10", "snmfwlp2026"):
            settings = [
                setting
                for record in records
                for setting in record["settings"]
                if setting["setting"] == setting_name
            ]
            if not settings:
                continue
            for arm in ARMS:
                for metric in METRICS:
                    values = np.asarray(
                        [setting["metrics"][arm][metric] for setting in settings],
                        dtype=float,
                    )
                    rows.append(
                        {
                            "dataset": path.stem,
                            "setting": setting_name,
                            "arm": arm,
                            "metric": metric,
                            "runs": len(values),
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        }
                    )
            observed = np.asarray(
                [setting["metrics"]["observed_graph"]["acc"] for setting in settings]
            )
            candidate = np.asarray(
                [setting["metrics"]["candidate_refinement"]["acc"] for setting in settings]
            )
            final = np.asarray(
                [setting["metrics"]["TRGNMF"]["acc"] for setting in settings]
            )
            accepted = np.asarray([bool(setting["accepted"]) for setting in settings])
            decisions.append(
                {
                    "dataset": path.stem,
                    "setting": setting_name,
                    "runs": len(settings),
                    "acceptance_rate": float(accepted.mean()),
                    "candidate_acc_gain": float((candidate - observed).mean()),
                    "final_acc_gain": float((final - observed).mean()),
                    "selected_k": dict(
                        sorted(Counter(str(setting["neighbors"]) for setting in settings).items())
                    ),
                    "maximum_sample_fraction_without_label_reach": float(
                        max(
                            setting["components"]["sample_fraction_without_label_reach"]
                            for setting in settings
                        )
                    ),
                }
            )

    with (args.output_dir / "metric_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "decision_summary.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
