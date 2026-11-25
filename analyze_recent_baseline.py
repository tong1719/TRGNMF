"""Compare TRGNMF and the public 2026 HLPNMF code on matched splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = ("acc", "nmi", "ari", "purity", "recall", "fscore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--hlpnmf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.hlpnmf_dir.glob("*.json")):
        dataset = path.stem
        hlp = {int(row["seed"]): row for row in json.loads(path.read_text(encoding="utf-8"))}
        trg = {
            int(row["seed"]): row
            for row in json.loads((args.validated_dir / f"{dataset}.json").read_text(encoding="utf-8"))
        }
        seeds = sorted(set(hlp).intersection(trg))
        for metric in METRICS:
            h = np.asarray([hlp[seed][metric] for seed in seeds], dtype=float)
            t = np.asarray([trg[seed]["selected"]["metrics"]["TRGNMF"][metric] for seed in seeds])
            difference = t - h
            try:
                p = float(wilcoxon(difference, zero_method="pratt").pvalue)
            except ValueError:
                p = 1.0
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "runs": len(seeds),
                    "TRGNMF_mean": t.mean(),
                    "TRGNMF_std": t.std(ddof=1),
                    "HLPNMF_mean": h.mean(),
                    "HLPNMF_std": h.std(ddof=1),
                    "mean_difference": difference.mean(),
                    "wilcoxon_p": p,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
