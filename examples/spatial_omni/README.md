# Spatial-Omni

[Paper](https://arxiv.org/abs/2606.10738) · [Checkpoints](https://huggingface.co/dieKarotte/Spatial-Omni) · [SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench)

Spatial-Omni answers questions about sounds and their spatial relationships. SO-Encoder extracts spatial features from four-channel first-order ambisonic (FOA) audio alongside the base model's audio encoder. This example includes SO-7B and SO-30B, with their SO and MIX checkpoints.

## Installation

After installing Auden, enter this directory:

~~~bash
cd examples/spatial_omni
~~~

Install the requirements for your model in its own environment.

SO-7B (Python 3.11):

~~~bash
pip install -r requirements.txt
~~~

SO-30B (Python 3.12, 80 GB GPU):

~~~bash
pip install -r requirements-so30b.txt
~~~

## Inference

Download a sample recording and ask a spatial question:

~~~bash
hf download dieKarotte/SO-Bench audio/test/foa_9975d396da0c90324197.wav \
  --repo-type dataset --local-dir data

python infer.py --audio data/audio/test/foa_9975d396da0c90324197.wav \
  --question "Which sound source is located to the front-right and above?"
~~~

The default model is SO-7B. Select another checkpoint with `--config`:

| Model | Config |
|---|---|
| SO-7B | `configs/so7b.yaml` |
| SO-7B-MIX | `configs/so7b_mix.yaml` |
| SO-30B | `configs/so30b.yaml` |
| SO-30B-MIX | `configs/so30b_mix.yaml` |

Weights are downloaded automatically from Hugging Face. To use local copies, pass `--checkpoint-dir` for the Spatial-Omni release and `--base-model` for the base model. Keep each adaptation checkpoint beside its `train_args.json`.

Input must be 16 kHz FOA with four channels and W in channel 0. The model uses the first 20 seconds. Coordinates follow DCASE: +x front, +y left, +z up.

For mono recordings, use a MIX config and add `--mono`. SO-7B uses learned null spatial tokens; SO-30B uses zero spatial tokens, matching its released evaluation path.

## Batch inference

Prepare a JSONL file with `qa_id` (or `id`), `audio_path`, and `question`. A `prompt` field can include answer choices and takes precedence over `question`.

~~~bash
python evaluate.py --config configs/so7b.yaml \
  --manifest qa.jsonl --audio-root data --limit 16 \
  --output outputs/predictions.jsonl
~~~

Predictions retain the input records and add a `prediction` field. Use `--dry-run` to check the inputs, or `--limit 0` to process the full manifest. Output paths must be new; completed predictions remain available if a run fails.

## License and citation

Code is licensed under Apache-2.0, with third-party notices in [NOTICE](NOTICE). Spatial-Omni weights use CC-BY-NC-SA-4.0. Base models and datasets retain their respective licenses.

~~~bibtex
@misc{zhu2026spatialomnispatialaudiounderstanding,
  title={Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding},
  author={Zhiyuan Zhu and Yixuan Chen and Yiwen Shao and Wenxiang Guo and Changhao Pan and Yu Zhang and Yuxiang Wang and Wei Liu and Houhua Zhang and Chengkuan Zeng and Wenbo Cheng and Yunxi Liu and Rui Yang and Steve Yves and Liefeng Bo and Zhou Zhao},
  year={2026},
  eprint={2606.10738},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
~~~
