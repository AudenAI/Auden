"""
One-time script: assemble a LALM checkpoint from pretrained components.

After running this script, load the model with standard HuggingFace APIs::

    model = AutoModelForCausalLM.from_pretrained(output_dir)
    processor = AutoProcessor.from_pretrained(output_dir)

Usage::

    python assemble_model.py \\
        --llm  Qwen/Qwen2-7B-Instruct \\
        --encoder openai/whisper-large-v3 \\
        --output_dir ./lalm_checkpoint
"""

import torch
from lalm_core.model import LALMConfig, LALMForConditionalGeneration, LALMProcessor
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def assemble_and_save(
    llm_name: str,
    encoder_name: str,
    output_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    projector_downsample_rate: int = 4,
):
    print(f"Loading LLM:     {llm_name}")
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch_dtype)

    print(f"Loading encoder: {encoder_name}")
    encoder, audio_dim = _load_encoder(encoder_name, torch_dtype)

    # Build processor first to obtain the modality_token_id after
    # the new <|audio|> token has been added to the vocabulary.
    print("Building processor ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
    processor = LALMProcessor(
        feature_extractor,
        tokenizer,
        encoder_name=_get_encoder_model_type(encoder),
        projector_downsample_rate=projector_downsample_rate,
    )

    # Resize LLM embeddings to cover the newly added audio token.
    llm.resize_token_embeddings(len(processor.tokenizer))

    text_dim = llm.config.hidden_size

    config = LALMConfig(
        text_config=llm.config,
        audio_config=encoder.config,
        audio_dim=audio_dim,
        text_dim=text_dim,
        projector_downsample_rate=projector_downsample_rate,
        audio_token_id=processor.tokenizer.convert_tokens_to_ids(processor.audio_token),
    )

    print("Assembling model ...")
    model = LALMForConditionalGeneration(
        config,
        language_model=llm,
        audio_tower=encoder,
    )
    # Projector starts randomly initialised and is learned during training.

    print(f"Saving to {output_dir} ...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Done.")

    return model, processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_encoder(
    model_name_or_path: str, torch_dtype: torch.dtype
) -> tuple[torch.nn.Module, int]:
    """
    Load an audio encoder and return ``(encoder_module, audio_dim)``.

    Do not rely on AutoModel here because some encoder checkpoints are not fully
    registered for generic auto loading. Use model-family-specific APIs instead.
    """
    hf_cfg = AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(hf_cfg, "model_type", "")

    if model_type == "whisper":
        from transformers.models.whisper.modeling_whisper import WhisperModel

        try:
            model = WhisperModel.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype
            )
            encoder = model.encoder
        except Exception:
            # Some checkpoints may contain encoder-only weights.
            from transformers.models.whisper.modeling_whisper import WhisperEncoder

            encoder = WhisperEncoder.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype
            )
        return encoder, int(encoder.config.d_model)

    if model_type in {"qwen2_5_omni", "qwen2_5_omni_audio_encoder"}:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniAudioEncoder,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        if model_type == "qwen2_5_omni":
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype
            )
            encoder = model.audio_tower
        else:
            encoder = Qwen2_5OmniAudioEncoder.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype
            )
        return encoder, int(encoder.config.output_dim)

    if model_type in {"qwen3_omni_moe", "qwen3_omni_moe_audio_encoder"}:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        if model_type == "qwen3_omni_moe":
            model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype
            )
            encoder = model.audio_tower
        else:
            encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype
            )
        return encoder, int(encoder.config.output_dim)

    raise ValueError(
        f"Unsupported encoder model_type={model_type!r}. "
        "Supported values: 'whisper', 'qwen2_5_omni', "
        "'qwen2_5_omni_audio_encoder', 'qwen3_omni_moe', "
        "'qwen3_omni_moe_audio_encoder'."
    )


def _get_encoder_model_type(encoder) -> str:
    hf_model_type = getattr(encoder.config, "model_type", "")
    supported = {
        "whisper",
        "qwen2_5_omni_audio_encoder",
        "qwen3_omni_moe_audio_encoder",
    }
    if hf_model_type not in supported:
        known = ", ".join(f'"{k}"' for k in sorted(supported))
        raise ValueError(
            f"Unsupported encoder model_type={hf_model_type!r}. "
            f"Supported values: {known}."
        )
    return hf_model_type


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Assemble LALM from pretrained components."
    )
    parser.add_argument(
        "--llm", required=True, help="HF model id or local path for the LLM"
    )
    parser.add_argument(
        "--encoder",
        required=True,
        help="HF model id or local path for the audio encoder",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where to save the assembled checkpoint"
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--projector_downsample_rate", default=4, type=int)
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    assemble_and_save(
        llm_name=args.llm,
        encoder_name=args.encoder,
        output_dir=args.output_dir,
        torch_dtype=dtype_map[args.dtype],
        projector_downsample_rate=args.projector_downsample_rate,
    )
