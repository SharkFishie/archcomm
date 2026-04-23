# archcomm
Active research on how agent architecture affect the structured properties of emergent communication in referential signalling games.

# archcomm

An empirical study of how agent architecture affects the structural properties of emergent communication in multi-agent referential games.

Two agents — a sender and a receiver — must develop a shared communication system from scratch to win a signaling game. No human language, no prior symbols. We vary only the agent architecture (LSTM, GRU, Transformer, MLP) 
and measure what kind of proto-language emerges in each case.

Built on top of [EGG](https://github.com/facebookresearch/EGG) 
(Facebook Research).

---

## Research question

Does the inductive bias of an agent's architecture shape the compositional structure of the language it develops?

---

## Setup

```bash
git clone https://github.com/yourusername/archcomm
cd archcomm
pip install -r requirements.txt
```

## Run a training experiment

```bash
python scripts/train.py --arch lstm --seed 42
python scripts/train.py --arch transformer --seed 42
```

## Run all conditions

```bash
bash scripts/run_all.sh
```

## Evaluate

```bash
python scripts/evaluate.py --results_dir results/
```

---

## Architectures compared

| Agent | Notes |
|---|---|
| LSTM | Baseline — most prior work uses this |
| GRU | Lighter recurrent baseline |
| Transformer | Main novel condition |
| MLP | Control — no sequential processing |

---

## Metrics

- **Topographic similarity** — does similar meaning → similar message?
- **Symbol entropy** — how evenly are symbols used?
- **Message length distribution** — do agents compress efficiently?
- **Generalisation** — does the language hold on unseen inputs?

---

## Status

🔬 Active research — experiments in progress

---

## Citation

If this work is useful to you, a citation will be available once the 
paper is published. Check back here.

---

## Author

Maria B. 
mariabusygina05@gmail.com

venv activation:
.venv\Scripts\activate