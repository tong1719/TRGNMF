"""Summarize revised parameter-sensitivity and runtime experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def sensitivity(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        groups = sorted({(record["axis"], float(record["value"])) for record in records})
        for axis, value in groups:
            selected = [
                record
                for record in records
                if record["axis"] == axis and float(record["value"]) == value
            ]
            observed = np.asarray(
                [record["metrics"]["observed_graph"]["acc"] for record in selected]
            )
            final = np.asarray([record["metrics"]["TRGNMF"]["acc"] for record in selected])
            rows.append(
                {
                    "dataset": path.stem,
                    "axis": axis,
                    "value": value,
                    "runs": len(selected),
                    "acceptance_rate": float(np.mean([record["accepted"] for record in selected])),
                    "observed_acc": float(observed.mean()),
                    "TRGNMF_acc": float(final.mean()),
                    "acc_gain": float((final - observed).mean()),
                }
            )
    if rows:
        with (output_dir / "sensitivity_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def runtime(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not records:
            continue
        observed_seconds = float(
            np.mean([record["observed_seconds"] for record in records])
        )
        trgnmf_seconds = float(
            np.mean([record["TRGNMF_seconds"] for record in records])
        )
        rows.append(
            {
                "dataset": path.stem,
                "runs": len(records),
                "observed_seconds": observed_seconds,
                "TRGNMF_seconds": trgnmf_seconds,
                "NMF_seconds": float(np.mean([record["NMF_seconds"] for record in records])),
                "absolute_overhead_seconds": float(
                    np.mean(
                        [
                            record["TRGNMF_seconds"] - record["observed_seconds"]
                            for record in records
                        ]
                    )
                ),
                "refinement_validation_seconds": float(
                    np.mean(
                        [
                            record["TRGNMF_seconds"]
                            - record["observed_seconds"]
                            - record["NMF_seconds"]
                            for record in records
                        ]
                    )
                ),
                # Match the ratio to the mean times reported in this row.
                "overhead_ratio": trgnmf_seconds / observed_seconds,
            }
        )
    if rows:
        with (output_dir / "runtime_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "sensitivity": sensitivity(args.sensitivity_dir, args.output_dir),
        "runtime": runtime(args.runtime_dir, args.output_dir),
    }
    (args.output_dir / "sensitivity_runtime_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

