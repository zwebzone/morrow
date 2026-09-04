# Morrow: anonymous training release

This directory is a compact reference implementation of the training method
described in the accompanying submission. It contains only the method's
trainable components, objectives, recurrent training loop, configuration, and
tests. It does not include evaluation results, internal infrastructure, or
benchmark data.

Morrow converts the native recurrent state produced while reading a session
into the persistent state used to initialize the next session. The pretrained
language model remains frozen. Training updates only learned write tokens, a
write-only LoRA, and a low-rank state carrier.

## Contents

- `morrow/backbone.py`: cold-KV GatedDeltaNet state injection and extraction.
- `morrow/lora.py`: learned write tokens and write-only LoRA.
- `morrow/carrier.py`: low-rank state reparameterization and stable carry.
- `morrow/objectives.py`: post-transition NLL, pre-transition NLL, and
  memory-off KL preservation.
- `morrow/trainer.py`: recurrent-depth curriculum and truncated BPTT.
- `configs/morrow_4b.yaml` and `configs/morrow_27b.yaml`: reference
  configurations matching the paper's architecture and loss coefficients;
  the optimization budget remains an explicit, editable field.
- `TECHNICAL.md`: equations, tensor conventions, and implementation mapping.

## Installation

Python 3.10 or newer and a CUDA build of PyTorch are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

The reference path targets a Transformers model whose configuration marks
GatedDeltaNet blocks as `linear_attention` and whose cache exposes
`recurrent_states`. The included configuration uses Qwen3.5-4B. The code is a
single-device reference implementation; distributed wrapping can be added
without changing the method.

## Training

Prepare the two JSONL files specified in [data/README.md](data/README.md), then
run:

```bash
python train.py --config configs/morrow_4b.yaml --device cuda:0
```

The same objective can be used for the paper's single-session initialization
by temporarily setting `max_unroll: 1` and `tbptt_horizon: 1`. Recurrent
curriculum training can then start from that writer/carrier checkpoint:

```bash
python train.py --config configs/morrow_4b.yaml --device cuda:0 \
  --initialize-from outputs/single-session/checkpoint-XXXXXXXX/morrow.safetensors
```

The second phase resets optimizer state but loads every trainable Morrow
parameter strictly; a changed architecture or adapter target set is rejected.

Metrics are appended to `metrics.jsonl`. Checkpoints contain only the Morrow
parameters in `morrow.safetensors`; frozen backbone weights are never copied.
The `training.complete` marker is created only after all configured steps
finish.

## Verification

```bash
pytest -q
python -m compileall -q morrow train.py
```

The tests cover the carrier equation, global norm-preserving SLERP, curriculum,
write-only adapter gating, configuration validation, and strict data loading.
