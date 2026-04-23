# Emergent Language & Architecture

Experiment comparing emergent communication across four sender/receiver architectures (LSTM, GRU, Transformer, MLP) on a referential game, measuring compositionality via topographic similarity.

## Setup

```bash
pip install -r requirements.txt
```

EGG requires a separate install if not on PyPI:

```bash
pip install git+https://github.com/facebookresearch/EGG.git
```

## Run a training

```bash
# From the repo root
python scripts/train.py --config configs/base_config.yaml --arch lstm
python scripts/train.py --config configs/base_config.yaml --arch gru
python scripts/train.py --config configs/base_config.yaml --arch transformer
python scripts/train.py --config configs/base_config.yaml --arch mlp
```

## Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint results/lstm/baseline/best_model.pt \
    --config configs/base_config.yaml \
    --arch lstm --split test
```

## Key metrics

| Metric | Description |
|---|---|
| Accuracy | Receiver top-1 accuracy on referential game |
| Topo ρ | Spearman correlation between meaning & message distances (compositionality proxy) |
| Message entropy | Shannon entropy over the message distribution |
| Effective vocab size | Symbols used with frequency > 0.1% |

## Architecture variants

| Key | Module |
|---|---|
| `lstm` | LSTM sender + LSTM receiver |
| `gru` | GRU sender + GRU receiver |
| `transformer` | Transformer encoder sender + Transformer encoder receiver |
| `mlp` | MLP sender + mean-pool MLP receiver (sequence-order ablation) |

## Config

Copy and edit `configs/base_config.yaml` for ablations. All CLI flags override config values.
