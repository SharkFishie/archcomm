# archcomm — development log
 
Chronological record of errors encountered during setup and experiment runs, with root causes and fixes applied.
 
---
 
## 1. `editdistance` fails to build on Windows
 
**when:** initial `pip install egg-lib` / `pip install git+https://github.com/facebookresearch/EGG.git`
 
**error:**
```
ERROR: Failed building wheel for editdistance
error: Microsoft Visual C++ 14.0 or greater is required.
```
 
**root cause:** `editdistance` is a C extension that must be compiled from source. Windows requires Microsoft Visual C++ Build Tools to compile C/C++ extensions. These were not installed.
 
**fix:** created a fake `editdistance` package in site-packages that wraps `python-Levenshtein`, which ships pre-built wheels for Windows:
 
```python
import site, os
path = site.getsitepackages()[0]
os.makedirs(os.path.join(path, 'editdistance'), exist_ok=True)
with open(os.path.join(path, 'editdistance', '__init__.py'), 'w') as f:
    f.write('from Levenshtein import distance\ndef eval(a, b): return distance(a, b)\n')
```
 
**files changed:** none in repo — patch applied to local venv and Colab environment at runtime.
 
---
 
## 2. EGG not importable after install
 
**when:** after `pip install git+https://github.com/facebookresearch/EGG.git --no-deps`
 
**error:**
```
ModuleNotFoundError: No module named 'egg'
```
 
**root cause:** `--no-deps` skipped all dependencies, so EGG installed but its dependencies (wandb, pandas, scikit-learn, etc.) were missing. EGG's `__init__.py` imports them immediately on load.
 
**fix:** installed all dependencies manually:
```bash
pip install wandb pandas pytest rich scikit-learn submitit timm torchvision dataclasses
```
 
---
 
## 3. EGG imports `editdistance` directly in source
 
**when:** after installing all dependencies
 
**error:**
```
ModuleNotFoundError: No module named 'editdistance'
```
 
**root cause:** `egg/core/language_analysis.py` has a hardcoded `import editdistance` at line 10. Even though EGG itself installed, this import fails because `editdistance` has no pre-built wheel. The fake package fix from issue 1 resolves this on subsequent setups.
 
**fix:** same fake package fix as issue 1, applied before importing egg.core.
 
---
 
## 4. EGG LSTM hidden state TypeError — Windows/local
 
**when:** first attempt to run `scripts/train.py --arch lstm`
 
**error:**
```
TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not tuple
```
at `egg/core/reinforce_wrappers.py` line 305.
 
**root cause:** EGG was written for an older PyTorch version. In newer PyTorch, LSTM hidden state is returned as a tuple `(h, c)`. EGG's code passes `prev_hidden[0]` directly to `torch.zeros_like()`, which expects a Tensor but gets a tuple.
 
**fix:** patched `reinforce_wrappers.py`:
```python
# before
torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
 
# after
torch.zeros_like(prev_hidden[0] if isinstance(prev_hidden, (tuple, list)) else prev_hidden) for _ in range(self.num_layers)
```
 
---
 
## 5. EGG LSTM hidden state TypeError — Colab (nested tuple)
 
**when:** running on Google Colab after applying fix from issue 4
 
**error:**
```
TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not tuple
```
same line, different cause.
 
**root cause:** on Colab's PyTorch version, `prev_hidden[0]` is itself a tuple `(h, c)` — one level deeper nesting than the Windows case.
 
**fix:** patched `reinforce_wrappers.py` more deeply — extracted `_h0` from `self.agent()` output before building `prev_hidden`:
 
```python
# before
prev_hidden = [self.agent(x, aux_input)]
prev_hidden.extend(
    [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
)
 
# after
_agent_out = self.agent(x, aux_input)
_h0 = _agent_out[0] if isinstance(_agent_out, (tuple, list)) else _agent_out
prev_hidden = [_h0]
prev_hidden.extend(
    [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
)
```
 
Applied automatically in Colab cell 2 via string replacement.
 
---
 
## 6. `ModuleNotFoundError: No module named 'agents'` when running train.py
 
**when:** running `python scripts/train.py` from repo root or from `emergent-lang-arch/`
 
**error:**
```
ModuleNotFoundError: No module named 'agents'
```
 
**root cause:** Python doesn't automatically add the current working directory to `sys.path` when running a script in a subdirectory. The `agents/` package is in `emergent-lang-arch/` but Python can't find it.
 
**fix:** set PYTHONPATH before running:
```bash
# local
$env:PYTHONPATH = "."
python scripts/train.py ...
 
# Colab
!PYTHONPATH=. python scripts/train.py ...
```
 
---
 
## 7. Receiver embedding TypeError — float tensor as token indices
 
**when:** after PYTHONPATH fix, first successful game forward pass attempt
 
**error:**
```
RuntimeError: Expected tensor for argument #1 'indices' to have scalar types: Long, Int; 
but got torch.FloatTensor instead
```
 
**root cause:** a misunderstanding of how EGG's `RnnReceiverDeterministic` works. EGG runs its own internal `RnnEncoder` on the incoming message tokens, then calls our receiver with the *encoded hidden state* (a float vector), not the raw token IDs. All four receiver `forward()` methods were calling `self.embed(message)` on this float hidden state, which is garbage input to an embedding layer.
 
**fix:** rewrote all four receiver cores to accept the pre-encoded hidden state directly and use it to score candidates, removing the embed/encode layers from the receiver entirely. This is the architecturally correct approach — EGG handles message encoding, our receiver just needs to score.
 
**files changed:** `agents/lstm_agent.py`, `agents/gru_agent.py`, `agents/transformer_agent.py`, `agents/mlp_agent.py`
 
**result:** accuracy jumped from ~20% (random chance) to ~45% by epoch 10, with a clear learning transition around epoch 7.
 
---
 
## 8. Training frozen at epoch 1, batch 1 on Colab
 
**when:** first Colab run with full base config
 
**symptom:** script printed `Epoch 1 batch 1/195 running...` then hung indefinitely (13+ minutes).
 
**root cause:** `base_config.yaml` had `device: "cpu"` while Colab has a T4 GPU. PyTorch was running 50k samples × 100 epochs entirely on CPU, which is extremely slow for REINFORCE training.
 
**fix:**
```bash
sed -i 's/device: "cpu"/device: "cuda"/' configs/base_config.yaml
```
 
Also created `configs/dev_config.yaml` with reduced scale for local testing:
- `n_train: 2000`, `n_val: 200`, `n_test: 200`
- `epochs: 20`, `batch_size: 64`
---
 
## 9. Accuracy stuck at ~20%, topo_rho NaN for full run
 
**when:** first complete 100-epoch run on Colab (after device fix)
 
**symptom:** accuracy never exceeded random chance (~20% for 5 candidates), topo_rho mostly NaN with a `ConstantInputWarning` — agents sending identical messages regardless of input.
 
**root cause:** the receiver fix from issue 7 had not yet been applied to this run. Agents appeared to converge to a degenerate solution (always send the same message) because receiver feedback was useless.
 
**fix:** applied receiver fix from issue 7, reran. Confirmed learning on dev config before proceeding to full run.
 
---
 
## current status
 
- pipeline runs end-to-end on Colab with GPU
- LSTM baseline confirmed learning (accuracy > random, topo_rho computable)
- ready to run full sweep: 4 architectures × 10 seeds