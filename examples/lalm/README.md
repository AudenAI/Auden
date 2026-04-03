# LALM

LALM is a multimodal language model that combines a pretrained audio encoder with a causal LLM via a learned projector. The model follows the standard HuggingFace `from_pretrained` / `save_pretrained` interface throughout.

Supported audio encoders: Whisper, Qwen2.5-Omni audio encoder, Qwen3-Omni audio encoder.

---

## Overview

```
audio waveform
     │
     ▼
Audio Encoder  (frozen or fine-tuned)
     │  packed features (C, T_total)
     ▼
LALMProjector  (learned)
     │  audio embeddings inserted into LLM token sequence
     ▼
Causal LLM     (frozen or fine-tuned)
     │
     ▼
text output
```

Training follows a two-stage recipe:
- **Stage 1** — freeze encoder + LLM, train projector only (`frozen_modules: [audio_tower, language_model]`)
- **Stage 2** — unfreeze all, full fine-tuning (`frozen_modules: []`)

---

## Step 1 — Build Model

Assemble a LALM checkpoint from a pretrained LLM and audio encoder. This is a one-time step that produces a self-contained HF checkpoint.

```bash
bash scripts/build_model.sh
```

Key options in `scripts/build_model.sh`:

| Variable | Description |
|---|---|
| `llm` | Path to pretrained LLM (e.g. Qwen2.5-7B-Instruct) |
| `encoder` | Path to audio encoder (e.g. Qwen3-Omni, Qwen2.5-Omni, Whisper) |
| `projector_downsample_rate` | Frame concat rate in projector (higher = fewer LLM audio tokens) |
| `output_dir` | Where to save the assembled checkpoint |

The assembled checkpoint can be loaded like any HF model:

```python
from lalm_core.model import LALMForConditionalGeneration, LALMProcessor

model = LALMForConditionalGeneration.from_pretrained(output_dir)
processor = LALMProcessor.from_pretrained(output_dir)
```

---

## Step 2 — Prepare Manifest

Each training/evaluation sample needs a `conversation` field attached to its Lhotse cut. Run `prepare_conversation.py` once per dataset split.

```bash
bash scripts/prepare_manifest.sh
```

Or run directly:

```bash
python prepare_conversation.py \
    --input_manifest /path/to/cuts.jsonl.gz \
    --output_manifest data/train/cuts_conversation.jsonl.gz \
    --tokenizer /path/to/llm \
    --instruction "Please transcribe speech."
```

Key options:

| Argument | Description |
|---|---|
| `--input_manifest` | Input Lhotse CutSet manifest |
| `--output_manifest` | Output manifest with `conversation` field added |
| `--tokenizer` | Tokenizer path (used to estimate token counts for batching) |
| `--instruction` | Optional user instruction appended to each sample |
| `--system` | Optional system prompt |

The prepared manifest stores two fields per cut:
- `cut.conversation` — structured message list (OpenAI chat format)
- `cut.rendered_conversation` — rendered chat string used directly during training

Register your prepared manifests in `configs/train_data_config.yaml` and `configs/valid_data_config.yaml`:

```yaml
- name: aishell1
  manifest: data/train/aishell1_cuts_conversation.jsonl.gz
```

---

## Step 3 — Train

```bash
bash scripts/train.sh
```

Multi-node example:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<host0> bash scripts/train.sh
```

Key options in `scripts/train.sh`:

| Variable | Description |
|---|---|
| `model_dir` | Path to checkpoint from Step 1 |
| `frozen_modules` | Comma-separated modules to freeze (e.g. `audio_tower,language_model`) |
| `mixed_precision` | `bf16` or `fp16` |
| `exp_name` | Experiment name; checkpoints saved to `exp/<exp_name>/` |

Key training config options (override via command line or `configs/train.yaml`):

| Config key | Description |
|---|---|
| `trainer.optimizer.lr` | Learning rate |
| `trainer.num_steps` | Total training steps |
| `trainer.grad_accum_steps` | Gradient accumulation steps |
| `trainer.valid_interval` | Validate every N steps |
| `trainer.save_every_n` | Save checkpoint every N validation intervals |
| `data.sampler.max_tokens` | Max LLM tokens per batch (controls batch size) |
| `data.feature` | Feature type: `whisper_v3_fbank` (128-dim) or `whisper_fbank` (80-dim) |

Checkpoints are saved to `exp/<exp_name>/checkpoint-{step}.pt`. The HF config and processor are saved once at the start of training to `exp/<exp_name>/hf/`.

To resume training, set `trainer.start_batch=<step>` to the checkpoint step you want to resume from.

---

## Step 4 — Evaluate

```bash
bash scripts/evaluate.sh
```

Or run directly:

```bash
python evaluate.py \
    exp_dir=./exp/my_experiment \
    checkpoint.iter=16000 \
    data.test_data_config=configs/valid_data_config.yaml \
    decoding_method=greedy_search
```

Key options:

| Config key | Description |
|---|---|
| `exp_dir` | Experiment directory containing checkpoints |
| `checkpoint.iter` | Load `checkpoint-{iter}.pt` |
| `checkpoint.epoch` | Load `epoch-{epoch}.pt` |
| `checkpoint.model_dir` | Load directly from a HF model directory (skip export) |
| `decoding_method` | `greedy_search` or `beam_search` |
| `num_beams` | Beam size (used when `decoding_method=beam_search`) |
| `dtype` | Inference dtype: `fp16`, `bf16`, or `fp32` |

On first run, the trainer checkpoint is exported to a HF checkpoint at `exp/<exp_dir>/export/iter-{iter}/` and reused on subsequent runs.

Results are written to `exp/<exp_dir>/<decoding_method>/`.
