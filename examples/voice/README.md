# Voice Multitask Model

This example trains a general-purpose voice encoder.

In our experiments, multi-task learning worked best — combining speaker, emotion, gender, and age classification helped the model learn more robust voice representations. 

This example shows how to set up that multi-task training with a shared audio encoder and multiple classification heads, as well as the its usage as a stand-alone voice encoder. 

## Data Preparation

See detailed data preparation and manifest format in [configs/README.md](configs/README.md)



## Training



```bash
# Train from scratch with default small encoder (emb_dim=512)
torchrun --nproc_per_node=1 train.py \
  exp_dir=./exp/voice_exp

# Train from scratch with custom encoder config
torchrun --nproc_per_node=1 train.py \
  exp_dir=./exp/voice_exp \
  model.encoder=/path/to/encoder_config

# Fine-tune from pretrained encoder (config + weights loaded automatically)
torchrun --nproc_per_node=1 train.py \
  exp_dir=./exp/voice_exp \
  model.pretrained_encoder=/path/to/pretrained/encoder

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 train.py \
  exp_dir=./exp/voice_exp \
  model.pretrained_encoder=/path/to/pretrained/encoder
```

**Note:** 
- `model.encoder`: Specifies encoder config path or model type (e.g., "zipformer"). Optional, defaults to small zipformer.
- `model.pretrained_encoder`: Loads pretrained encoder weights AND config from a directory. Use this for fine-tuning.

## Evaluation


```bash
# From averaged checkpoints
python evaluate.py \
  exp_dir=./exp/voice_test \
  checkpoint.iter=20000 \
  checkpoint.avg=5 

# From a specific pretrained model
python evaluate.py \
  checkpoint.filename=/path/auden_encoder_voice

```

## Inference Guide

Two ways to use the trained voice model: **full model** for classification tasks, or **encoder-only** for speaker verification and various voice-related tasks (e.g., diarization, retrival, captioning, LLM-QA).

---

### 1. Full Model: Classification Tasks

Generate predictions for speaker ID, emotion, gender, and age.

```python
from model import VoiceMultitaskModel

# Load model
model = VoiceMultitaskModel.from_pretrained("./exp/voice_test/averaged-iter-1000-avg-0.pt").eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate predictions for each task
audio_file = "/path/to/audio.wav"

# Task options: "id", "emotion", "gender", "age"
labels, logits, probs = model.generate(input=[audio_file], task="emotion", topk=3)

print(f"Top predictions: {labels[0]}")
print(f"Probabilities: {probs[0]}")
```

**Expected Output:**
```
⚠️  Speaker identification is a closed-set task. The model will output the most likely speaker seen in training data. If you wish to do speaker verification (open-set), directly use the encoder to compare embedding distance/similarity.

🎵 Audio 1: sample.wav
----------------------------------------
  Speaker ID: Ses04F(0.434), Ses04M(0.173), 1036(0.138)
     Emotion: angry(0.752), frustrated(0.155), excited(0.041)
      Gender: male(1.000), female(0.000)
         Age: young adult(0.722), middle-age adult(0.211), senior(0.066)
```

**Note:** When `task="id"`, you'll see a warning that speaker ID is a closed-set task.

---

### 2. Encoder-Only: Speaker Verification and More

Extract embeddings and compare speakers using cosine similarity.

```python
from model import VoiceMultitaskModel
import torch
import torch.nn.functional as F

# Load model and extract encoder
full_model = VoiceMultitaskModel.from_pretrained("./exp/voice_test/averaged-iter-1000-avg-0.pt").eval()
encoder = full_model.encoder
encoder = encoder.to("cuda" if torch.cuda.is_available() else "cpu")

# Extract embeddings for 2 audio files
audio_files = ["/path/to/audio1.wav", "/path/to/audio2.wav"]
embeddings_list = []

for audio_file in audio_files:
    # Extract features and forward through encoder
    x, x_lens = encoder.extract_feature([audio_file])
    x, x_lens = x.to(device), x_lens.to(device)
    
    with torch.no_grad():
        encoder_output = encoder(x, x_lens)
        frame_embeddings = encoder_output["encoder_out"]  # [B, T, D]
        
        ## Below: Optionally for Speaker Verification
        # Global average pooling
        T = frame_embeddings.size(1)
        mask = (torch.arange(T, device=device).unsqueeze(0) < x_lens.unsqueeze(1)).unsqueeze(-1).float()
        utterance_embedding = (frame_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        
        embeddings_list.append(utterance_embedding)

# Stack and normalize embeddings
embeddings = torch.cat(embeddings_list, dim=0)  # [N, D]
embeddings = F.normalize(embeddings, p=2, dim=-1)

# Calculate cosine similarity
similarity = torch.matmul(embeddings[0], embeddings[1])
print(f"Cosine similarity: {similarity:.4f}")
print(f"Same speaker: {'✅ YES' if similarity >= 0.5 else '❌ NO'}")
```

**Expected Output:**
```
🎵 Audio 1:
   Frame embeddings shape: torch.Size([1, 97, 768])
   Utterance embedding shape: torch.Size([1, 768])

🎵 Audio 2: 
   Frame embeddings shape: torch.Size([1, 138, 768])
   Utterance embedding shape: torch.Size([1, 768])

Cosine similarity: 0.7234
Same speaker: ✅ YES
```

## Usage of the Pretrained Auden-Voice Encoder
Model: https://huggingface.co/AudenAI/auden-encoder-voice

```
from auden.auto.auto_model import AutoModel
encoder = AutoModel.from_pretrained("AudenAI/auden-encoder-voice")
```

### Performance
| Task - Dataset   | Metric (deault Acc)  |
|----------|--------|
| Speaker Identification - Vox2  | 95.25 % |
| Speaker Verification - Vox1-o  | EER 3% |
| Speaker Diarization - Voxconverse  | DER 17% |
| Age classification   - CREMA-D    | 93.91 % |
| Gender classification - CREMA-D        | 99.72% |
| Gender classification - RAVDESS        | 100% |
| Emotion classification - CREMA-D        | 83.99% |
| Emotion classification - RAVDESS       | 89.71% |
| Audio2Text retrieval - Paraspeechcaps       | R@1 63.31 |
| Text2Audio retrieval - Paraspeechcaps       | R@1 61.69 |
| LLM-QA Emotion - AirBench-MELD       | 27.23% |
| LLM-QA Emotion - AirBench-IEMOCAP       | 84.70% |
| LLM-QA Gender - AirBench-MELD       | 81.58% |
| LLM-QA Gender - AirBench-CommonVoice       | 93.15% |
| LLM-QA Age - AirBench-CommonVoice       | 58.27% |

## Citation

If you use this model in your research, please cite:

```bibtex
@inproceedings{huo2026auden,
  title     = {Auden-Voice: General-Purpose Voice Encoder for Speech and Language Understanding},
  author    = {Huo, Mingyue and Tseng, Wei-Cheng and Shao, Yiwen and Zhang, Hao and Yu, Dong},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

