# Auden: Audio & Multimodal Understanding Research Toolbox

A comprehensive toolbox for audio & multimodal understanding tasks including ASR, CLAP, audio captioning, speaker identification, speech-llm and more.

## Quick Start

Before installing Auden:
- Install PyTorch and Torchaudio following the official instructions: https://pytorch.org/get-started/locally/
- If you plan to use Zipformer/ASR/TTA, install a k2 wheel matching your PyTorch and CUDA versions. For example (torch==2.7.1, CUDA 11.8): `pip install k2==1.24.4.dev20250714+cuda11.8.torch2.7.1 -f https://k2-fsa.github.io/k2/cuda.html`. See the [k2 CUDA wheel matrix](https://k2-fsa.github.io/k2/cuda.html) and the [k2 installation guide](https://k2-fsa.github.io/k2/).

```bash
git clone https://github.com/AudenAI/Auden.git
cd Auden
pip install -e .
```
Some [examples/](examples/) may have extra installation requirements. Please refer to the [examples/](examples/) READMEs for details.

## Features

- 🎯 Multiple foundation audio tasks: ASR, captioning, contrastive learning, audio classification, speaker identification etc.
- 🤖 Multimodal LLM support (e.g. speech-LLM, asr-llm)
- 🚀 Pre-trained models&encoders with huggingface support and easy fine-tuning
- 🔧 Modular design for custom workflows
- 📊 Comprehensive evaluation metrics

## Usage

### Available Built-in Model Types from `AutoModel`

```python
from auden.auto import list_available_models
print(list_available_models())
```

### Custom Model Registration

**Important**: If you want to load custom models with `AutoModel`, you must register them first:

```python
from auden.auto import register_model, register_config

# Register your custom model and config
register_model("my-model", "examples.my_model.model", "MyModel")
register_config("my-model", "examples.my_model.config", "MyConfig")

# Now you can use it with AutoModel
from auden.auto import AutoModel
model = AutoModel.from_pretrained("path/to/my-model")
```

**Note**: If you don't want to use `AutoModel`, you can always skip this step and use your own way to load.

### Loading Models

Auden provides a HuggingFace-like interface for loading models:

```python
from auden.auto import AutoModel

# Load from HuggingFace Hub
model = AutoModel.from_pretrained("your-org/your-model")

# Load from local checkpoint
model = AutoModel.from_pretrained("path/to/model")

# Load from configuration (creates an EMPTY model)
from auden.auto import AutoConfig
config = AutoConfig.from_pretrained("path/to/config_or_model_dir")

# Important: from_config(...) constructs an EMPTY model (random init) from the config only
# It does NOT load weights. To load weights, use from_pretrained(...)
model = AutoModel.from_config(config)
```

### Loading Configurations

```python
from auden.auto import AutoConfig

# Load from various sources
config = AutoConfig.from_pretrained("your-org/model")
config = AutoConfig.from_pretrained("path/to/config.json")

# Create config for specific model type
config = AutoConfig.for_model("zipformer", hidden_size=512)
```

### Loading Tokenizers

```python
from auden.auto import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-org/tokenizer")
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
```

## Examples

Check [examples/](examples/) for task-specific tutorials.

## License
The code and weights in this repository are released under the [LICENSE](LICENSE) file.
This repository also includes a [NOTICE](NOTICE) file with third-party attributions (e.g., Transformers, Icefall).

## Acknowledgements
- This project draws inspiration from the design and user experience of [Hugging Face Transformers](https://github.com/huggingface/transformers), especially around `Auto*` APIs and configuration patterns.
- Many ASR components and utilities (e.g., Zipformer encoder variants, WER tooling) are adapted from or inspired by [k2-fsa/icefall](https://github.com/k2-fsa/icefall). We thank the Icefall authors and contributors for their excellent work and open-source spirit.
- Parts of the data pipeline and dataset handling build upon [Lhotse](https://github.com/lhotse-speech/lhotse). We thank the Lhotse authors and contributors for their work and open-source efforts.
