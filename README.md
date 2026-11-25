# TRGNMF

Code and experiment records for *Reliability-gated graph refinement with nonnegative matrix factorization for semi-supervised clustering*.

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the datasets listed in `data/README.md` under `data/spnmf_anchor/` and `data/image_benchmarks/`. Run the main experiment with:

```bash
python run_validated_gate.py all --data-dir data/spnmf_anchor
```

The experiment records used in the paper are provided in `results.zip`.

