from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from auden.auto.auto_model import AutoModel
from auden.models.lalm.model import (
    CHAT_TEMPLATE,
    IGNORE_TOKEN_ID,
    EncoderProjector,
    LalmModel,
    compute_accuracy,
)

# Qwen-3 chat template
CHAT_TEMPLATE_3 = """{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"""


class TagSpeechBaseModel(LalmModel):
    """TagSpeech base model with dual audio encoders and separate projectors.

    Architecture:
        Audio Input
            ↓
        ┌─────────────┬─────────────┐
        │             │             │
    Semantic Encoder  Voice Encoder
        │             │
        ↓             ↓
    Projector1    Projector2
        │             │
        ↓             ↓
    [B,L1,D_llm]  [B,L2,D_llm]
        │             │
        └─────┬───────┘
              │
        Text: "text <|AUDIO|> speaker <|AUDIO|>"
              │             │
              ↓             ↓
        Semantic Embedding Voice Embedding
              │             │
              └─────┬───────┘
                    │
                  LLM
    """

    def __init__(self, config, tokenizer):
        # First initialize parent class (creates semantic encoder, llm, etc.)
        super().__init__(config, tokenizer)

        # Note: parent class has created self.audio_encoder (semantic encoder)
        # Now we add voice encoder

        # Handle voice_encoder_config (could be dict or object)
        voice_config = config.voice_encoder_config
        if isinstance(voice_config, dict):
            self.voice_encoder_type = voice_config["model_type"]
        else:
            self.voice_encoder_type = voice_config.model_type

        # Create voice encoder
        # Convert dict to config if needed
        if isinstance(voice_config, dict):
            from auden.auto.auto_config import AutoConfig

            # Don't pass model_type twice - it's already in voice_config
            voice_config_copy = dict(voice_config)
            model_type = voice_config_copy.pop("model_type")
            voice_config_obj = AutoConfig.for_model(model_type, **voice_config_copy)
        else:
            voice_config_obj = voice_config

        self.voice_encoder = AutoModel.from_config(voice_config_obj)
        self.voice_encoder_dim = self.voice_encoder.encoder_out_dim

        # Delete parent's encoder_projector, create two independent projectors
        del self.encoder_projector

        # Create semantic projector
        self.semantic_projector = EncoderProjector(
            self.audio_encoder_dim,  # 768
            self.llm.config.hidden_size,  # D_llm
            config.semantic_projector_ds_rate,
        )

        # Create voice projector
        self.voice_projector = EncoderProjector(
            self.voice_encoder_dim,  # 768
            self.llm.config.hidden_size,  # D_llm
            config.voice_projector_ds_rate,
        )

    def _get_chat_template(self):
        """Auto-select chat template based on LLM model type."""
        # Check LLM config for model type
        llm_config = self.llm.config

        # Check for Qwen-3 models
        if hasattr(llm_config, "architectures") and llm_config.architectures:
            if any("Qwen3" in arch for arch in llm_config.architectures):
                return CHAT_TEMPLATE_3

        # Check for Qwen-2 models
        if hasattr(llm_config, "model_type"):
            if (
                "qwen2" in llm_config.model_type.lower()
                or "qwen-2" in llm_config.model_type.lower()
            ):
                return CHAT_TEMPLATE

        # Check for Qwen-3 in model_type
        if hasattr(llm_config, "model_type"):
            if (
                "qwen3" in llm_config.model_type.lower()
                or "qwen-3" in llm_config.model_type.lower()
            ):
                return CHAT_TEMPLATE_3

        # Default to original template
        return CHAT_TEMPLATE

    def forward_audio_features(self, x: torch.Tensor, x_lens: torch.Tensor):
        """Forward pass through dual encoders and projectors.

        Returns:
            semantic_features: [B, L1, D_llm] - for first audio token
            voice_features: [B, L2, D_llm] - for second audio token
            semantic_lens: [B] - lengths for semantic features
            voice_lens: [B] - lengths for voice features
        """
        return self._forward_dual_audio_features(x, x_lens)

    def _forward_dual_audio_features(self, x: torch.Tensor, x_lens: torch.Tensor):
        """Internal method to extract features from dual audio encoders."""
        # ========================================
        # 1. Extract features from both encoders
        # ========================================
        # Semantic encoder
        semantic_output = self.audio_encoder(x, x_lens)
        semantic_outs = semantic_output["encoder_out"]
        semantic_feature_lens = semantic_output["encoder_out_lens"]

        # Voice encoder
        voice_output = self.voice_encoder(x, x_lens)
        voice_outs = voice_output["encoder_out"]
        voice_feature_lens = voice_output["encoder_out_lens"]

        # ========================================
        # 2. Dual projectors
        # ========================================
        # Semantic projector
        semantic_features = self.semantic_projector(semantic_outs).to(torch.float16)
        semantic_lens = semantic_feature_lens // self.config.semantic_projector_ds_rate

        # Voice projector
        voice_features = self.voice_projector(voice_outs).to(torch.float16)
        voice_lens = voice_feature_lens // self.config.voice_projector_ds_rate

        return semantic_features, voice_features, semantic_lens, voice_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        messages: List[List[Dict[str, str]]],
        max_length: int = 384,
    ):
        """Override parent forward method to handle dual audio encoders."""
        # Get features from dual encoders
        semantic_features, voice_features, semantic_lens, voice_lens = (
            self.forward_audio_features(x, x_lens)
        )

        # Pass both features as a tuple for dual encoder preprocessing
        audio_features = (semantic_features, voice_features, semantic_lens, voice_lens)

        # Use the dual encoder preprocessing
        input_ids, inputs_embeds, attention_mask, labels = (
            self.preprocess_text_and_audio(
                messages,
                audio_features=audio_features,
                audio_feature_lens=None,  # Not used in dual mode
                max_length=max_length,
                is_training=True,
                tag_audio_boundary=self.config.tag_audio_boundary,
            )
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )
        with torch.no_grad():
            preds = torch.argmax(outputs.logits, -1)
            acc = compute_accuracy(
                preds[:, :-1], labels[:, 1:], ignore_label=IGNORE_TOKEN_ID
            )
        return outputs, acc

    def preprocess_text_and_audio(
        self,
        messages,
        audio_features=None,
        audio_feature_lens=None,
        max_length=192,
        tag_audio_boundary=False,
        is_training=False,
    ):
        """Override parent method to handle dual encoder tokens."""
        # For dual encoder model, we need to handle two separate audio features
        if audio_features is not None and len(audio_features) == 4:
            # audio_features is a tuple: (semantic_features, voice_features, semantic_lens, voice_lens)
            semantic_features, voice_features, semantic_lens, voice_lens = (
                audio_features
            )
            return self._preprocess_dual_audio_tokens(
                messages=messages,
                semantic_features=semantic_features,
                voice_features=voice_features,
                semantic_lens=semantic_lens,
                voice_lens=voice_lens,
                max_length=max_length,
                tag_audio_boundary=tag_audio_boundary,
                is_training=is_training,
            )
        else:
            # Fallback to parent method for single audio token
            return super().preprocess_text_and_audio(
                messages,
                audio_features,
                audio_feature_lens,
                max_length,
                tag_audio_boundary,
                is_training,
            )

    def _preprocess_dual_audio_tokens(
        self,
        messages,
        semantic_features,
        voice_features,
        semantic_lens,
        voice_lens,
        max_length=128,
        tag_audio_boundary=False,
        is_training=False,
    ):
        """Handle dual encoder tokens preprocessing.

        This method processes messages with TWO <|AUDIO|> tokens:
        - First <|AUDIO|> is replaced with semantic_features
        - Second <|AUDIO|> is replaced with voice_features
        """
        from auden.models.lalm.utils import IGNORE_TOKEN_ID

        batch_size = len(messages)

        # Get audio token id
        audio_token_id = self.tokenizer.convert_tokens_to_ids(self.config.audio_token)
        if audio_token_id is None or audio_token_id < 0:
            raise ValueError(
                f"audio_token '{self.config.audio_token}' is not in the tokenizer vocabulary."
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set.")

        # Auto-select chat template based on LLM model type
        chat_template = self._get_chat_template()

        # Training vs inference switches
        add_generation_prompt = not is_training
        prepare_label = is_training

        device = semantic_features.device

        # Prepare audio features
        semantic_max_len = int(semantic_lens.max().item())
        voice_max_len = int(voice_lens.max().item())
        semantic_features = semantic_features[:, :semantic_max_len]
        voice_features = voice_features[:, :voice_max_len]

        # Build per-sample ids by expanding TWO audio tokens
        semantic_lens_list = semantic_lens.tolist()
        voice_lens_list = voice_lens.tolist()
        input_ids_list = []
        max_input_len = 0

        for i, msg in enumerate(messages):
            # Apply chat template
            try:
                template_kwargs = dict(
                    tokenize=False,
                    chat_template=chat_template,
                    add_generation_prompt=add_generation_prompt,
                    padding="do_not_pad",
                    truncation=False,
                )
                if max_length is not None:
                    template_kwargs["max_length"] = max_length
                s = self.tokenizer.apply_chat_template(
                    msg,
                    **template_kwargs,
                )
            except Exception:
                # Fallback: join roles/contents simply
                s = "".join([f"{turn['role']}\n{turn['content']}\n" for turn in msg])

            ids = self.tokenizer.encode(s, add_special_tokens=False)

            # Find positions of the TWO audio tokens
            try:
                # Find first audio token
                first_audio_pos = ids.index(audio_token_id)

                # Find second audio token (search after the first one)
                try:
                    second_audio_pos = ids.index(audio_token_id, first_audio_pos + 1)
                except ValueError:
                    # Only one audio token found, use it for semantic only
                    logging.warning(
                        f"Sample {i}: Expected 2 audio tokens, but found only 1. "
                        f"Using it for semantic features only."
                    )
                    L1 = int(semantic_lens_list[i])
                    out = (
                        ids[:first_audio_pos]
                        + [audio_token_id] * L1
                        + ids[first_audio_pos + 1 :]
                    )
                    if max_length is not None:
                        out = out[:max_length]
                    input_ids_list.append(out)
                    if len(out) > max_input_len:
                        max_input_len = len(out)
                    continue

                # Expand both audio tokens
                L1 = int(semantic_lens_list[i])  # Length for semantic
                L2 = int(voice_lens_list[i])  # Length for voice

                # Build output ids:
                # [prefix] + [audio_token_id] * L1 + [middle] + [audio_token_id] * L2 + [suffix]
                out = (
                    ids[:first_audio_pos]  # prefix
                    + [audio_token_id] * L1  # expanded semantic tokens
                    + ids[first_audio_pos + 1 : second_audio_pos]  # middle part
                    + [audio_token_id] * L2  # expanded voice tokens
                    + ids[second_audio_pos + 1 :]  # suffix
                )

            except ValueError:
                # No audio token found at all
                logging.warning(f"Sample {i}: No audio token found in message.")
                out = ids

            if max_length is not None:
                out = out[:max_length]
            input_ids_list.append(out)
            if len(out) > max_input_len:
                max_input_len = len(out)

        # Pad ids
        if self.tokenizer.padding_side == "right":
            padded = [
                ids + [pad_id] * (max_input_len - len(ids)) for ids in input_ids_list
            ]
        else:
            padded = [
                [pad_id] * (max_input_len - len(ids)) + ids for ids in input_ids_list
            ]

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        padding_mask = input_ids == pad_id

        # Get text embeddings
        input_embeds = self.llm.get_input_embeddings()(input_ids)

        # Replace audio token embeddings with actual audio features
        input_embeds = input_embeds.clone()  # avoid in-place on view
        semantic_features = semantic_features.to(input_embeds.dtype)
        voice_features = voice_features.to(input_embeds.dtype)
        hidden_size = input_embeds.size(-1)

        if (
            semantic_features.size(-1) != hidden_size
            or voice_features.size(-1) != hidden_size
        ):
            raise ValueError(
                f"Audio feature dim mismatch: semantic={semantic_features.size(-1)}, "
                f"voice={voice_features.size(-1)}, LLM hidden={hidden_size}"
            )

        # Replace each sample's audio tokens
        # The structure is: [prefix] + [audio] * L1 + [middle] + [audio] * L2 + [suffix]
        # We need to find these two groups and replace them separately
        for i in range(batch_size):
            # Find all positions of audio tokens
            pos = (input_ids[i] == audio_token_id).nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue

            L1 = int(semantic_lens[i].item())  # semantic length
            L2 = int(voice_lens[i].item())  # voice length

            # Find the gap between the two groups
            # The gap is where pos[j+1] - pos[j] > 1 (not consecutive)
            if pos.numel() == 1:
                # Only one audio token, use for semantic
                input_embeds[i, pos[0] : pos[0] + 1] = semantic_features[i, :1]
                continue

            # Find the break point (where tokens are not consecutive)
            gaps = pos[1:] - pos[:-1]
            gap_idx = (gaps > 1).nonzero(as_tuple=False)

            if gap_idx.numel() == 0:
                # All tokens are consecutive, shouldn't happen but handle it
                # Assume first L1 are semantic, rest are voice
                K1 = min(L1, pos.numel())
                input_embeds[i, pos[:K1]] = semantic_features[i, :K1]
                if pos.numel() > L1:
                    K2 = min(L2, pos.numel() - L1)
                    input_embeds[i, pos[L1 : L1 + K2]] = voice_features[i, :K2]
            else:
                # Found gap, split into two groups
                split_idx = (
                    gap_idx[0].item() + 1
                )  # Index in pos array where second group starts

                # Group 1: semantic features
                group1_pos = pos[:split_idx]
                K1 = min(L1, group1_pos.numel())
                if K1 > 0:
                    input_embeds[i, group1_pos[:K1]] = semantic_features[i, :K1]
                if K1 < group1_pos.numel():
                    # Warn about unfilled tokens
                    logging.warning(
                        f"Sample {i}: Group1 has {group1_pos.numel()} tokens but only {K1} semantic features. "
                        f"Remaining {group1_pos.numel() - K1} tokens will use default embeddings."
                    )

                # Group 2: voice features
                group2_pos = pos[split_idx:]
                K2 = min(L2, group2_pos.numel())
                if K2 > 0:
                    input_embeds[i, group2_pos[:K2]] = voice_features[i, :K2]
                if K2 < group2_pos.numel():
                    # Warn about unfilled tokens
                    logging.warning(
                        f"Sample {i}: Group2 has {group2_pos.numel()} tokens but only {K2} voice features. "
                        f"Remaining {group2_pos.numel() - K2} tokens will use default embeddings."
                    )

        # Prepare labels and attention mask
        if prepare_label:
            labels = input_ids.clone().to(torch.long)
            labels[padding_mask] = IGNORE_TOKEN_ID

            # Ignore prompt up to assistant start marker
            try:
                assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
                im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                if (
                    assistant_token_id is not None
                    and assistant_token_id != self.tokenizer.unk_token_id
                    and im_start_id is not None
                ):
                    rows, cols = torch.where(input_ids == assistant_token_id)
                    for r, c in zip(rows.tolist(), cols.tolist()):
                        if c > 1 and input_ids[r, c - 1].item() == im_start_id:
                            labels[r, : c + 2] = IGNORE_TOKEN_ID
            except Exception:
                pass
        else:
            labels = None

        attention_mask = (~padding_mask).to(input_embeds.dtype)

        return input_ids, input_embeds, attention_mask, labels

    def generate(self, input, messages, max_length=None, **kwargs):
        """Override parent generate to handle dual encoder features."""
        if input is not None:
            if isinstance(input, tuple) and len(input) == 2:
                x, x_lens = input
            else:
                x, x_lens = self.audio_encoder.extract_feature(input)

            # Ensure features are on the same device as LLM
            x = x.to(self.llm.device)
            x_lens = x_lens.to(self.llm.device)

            # Get features from dual encoders
            semantic_features, voice_features, semantic_lens, voice_lens = (
                self.forward_audio_features(x, x_lens)
            )
            # Pack as tuple for preprocessing
            audio_features = (
                semantic_features,
                voice_features,
                semantic_lens,
                voice_lens,
            )
        else:
            audio_features = None

        # Enforce left padding for batched generation
        self.tokenizer.padding_side = "left"
        preprocess_max_length = max_length
        if preprocess_max_length is None:
            preprocess_max_length = getattr(self.config, "inference_max_length", None)

        input_ids, inputs_embeds, attention_mask, _ = self.preprocess_text_and_audio(
            messages,
            audio_features=audio_features,
            audio_feature_lens=None,  # Not used in dual mode
            max_length=preprocess_max_length,
            is_training=False,
            tag_audio_boundary=self.config.tag_audio_boundary,
        )
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            **kwargs,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    @classmethod
    def from_pretrained(
        cls, model_path: str, *, map_location: str | torch.device = "cpu", **kwargs
    ) -> "TagSpeechBaseModel":
        """Load a pretrained TagSpeechBaseModel.

        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
            map_location: Device to load the model on
            **kwargs: Additional arguments passed to model constructor (e.g., digit_embeddings)
        """
        import json
        import os

        from model_config import TagSpeechBaseConfig, TagSpeechConfig
        from safetensors.torch import load_file as safe_load_file
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer as HFTokenizer

        from auden.auto.auto_model import AutoModel

        # Support loading from HuggingFace Hub
        if not os.path.exists(model_path):
            model_path = AutoModel._download_from_hub(model_path)

        # Handle different model path formats
        if model_path.endswith((".pt", ".safetensors")):
            # Direct checkpoint file
            model_dir = os.path.dirname(model_path)
            weight_path = model_path
        else:
            # Directory path - look for weights
            model_dir = model_path
            weight_path = None
            for ext in [".pt", ".safetensors"]:
                candidate = os.path.join(model_path, f"model{ext}")
                if os.path.exists(candidate):
                    weight_path = candidate
                    break
            if weight_path is None:
                # Try to find any .pt or .safetensors file
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        if file.endswith((".pt", ".safetensors")):
                            weight_path = os.path.join(root, file)
                            break
                    if weight_path is not None:
                        break
            if weight_path is None:
                model_dir, _ = os.path.split(model_path)
                weight_path = model_path

        # Load config from JSON directly (avoid AutoConfig which doesn't recognize this model_type)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Determine which config class to use based on model_type
        model_type = config_dict.get("model_type", "tagspeech")
        if model_type == "tagspeech":
            config = TagSpeechConfig(**config_dict)
            is_tagspeech_model = True
        elif model_type == "tagspeech-base":
            config = TagSpeechBaseConfig(**config_dict)
            is_tagspeech_model = False
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Expected 'tagspeech' or 'tagspeech-base'"
            )

        tokenizer = HFTokenizer.from_pretrained(model_dir, use_fast=False)

        # Load digit embeddings for TagSpeech model if not provided
        if is_tagspeech_model and "digit_embeddings" not in kwargs:
            digit_emb_path = os.path.join(model_dir, "digit_embeddings.pt")
            if os.path.exists(digit_emb_path):
                logging.info(
                    f"[TagSpeech.from_pretrained] Loading digit embeddings from {digit_emb_path}"
                )
                kwargs["digit_embeddings"] = cls.load_digit_embeddings(digit_emb_path)
            else:
                logging.warning(
                    f"[TagSpeech.from_pretrained] digit_embeddings.pt not found in {model_dir}. "
                    "Model initialization may fail if digit_embeddings are required."
                )

        # Create model
        model = cls(config, tokenizer, **kwargs)

        # Load weights if present
        if weight_path and os.path.exists(weight_path):
            logging.info(
                f"[TagSpeechBaseModel.from_pretrained] Loading weights from {weight_path}"
            )
            ext = os.path.splitext(weight_path)[1].lower()
            if ext == ".safetensors":
                device_arg = (
                    str(map_location)
                    if isinstance(map_location, torch.device)
                    else map_location
                )
                state_dict = safe_load_file(weight_path, device=device_arg)
            else:
                state_dict = torch.load(weight_path, map_location=map_location)

            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )

        # Load projectors from projectors.pt (contains semantic_projector and voice_projector)
        projectors_path = os.path.join(model_dir, "projectors.pt")
        if os.path.exists(projectors_path):
            logging.info(
                f"[TagSpeechBaseModel] Loading projectors from {projectors_path}"
            )
            projector_state = torch.load(projectors_path, map_location=map_location)

            # Handle different checkpoint formats
            if isinstance(projector_state, dict) and "state_dict" in projector_state:
                projector_state = projector_state["state_dict"]
            elif isinstance(projector_state, dict) and "model" in projector_state:
                projector_state = projector_state["model"]

            # Extract semantic_projector and voice_projector weights
            semantic_state = {
                k.replace("semantic_projector.", ""): v
                for k, v in projector_state.items()
                if k.startswith("semantic_projector.")
            }
            voice_state = {
                k.replace("voice_projector.", ""): v
                for k, v in projector_state.items()
                if k.startswith("voice_projector.")
            }

            if semantic_state:
                model.semantic_projector.load_state_dict(semantic_state, strict=True)
                logging.info(f"[TagSpeechBaseModel] Loaded semantic_projector")
            if voice_state:
                model.voice_projector.load_state_dict(voice_state, strict=True)
                logging.info(f"[TagSpeechBaseModel] Loaded voice_projector")

        # Load audio_encoder from subdirectory (if excluded from checkpoint)
        audio_encoder_path = os.path.join(model_dir, "audio_encoder")
        if os.path.isdir(audio_encoder_path):
            try:
                ae = AutoModel.from_pretrained(audio_encoder_path)
                model.audio_encoder.load_state_dict(ae.state_dict(), strict=True)
                logging.info(
                    f"[TagSpeechBaseModel] Loaded audio_encoder from {audio_encoder_path}"
                )
            except Exception as e:
                logging.warning(
                    f"[TagSpeechBaseModel] Failed to load audio_encoder: {e}"
                )

        # Load voice_encoder from subdirectory (if excluded from checkpoint)
        voice_encoder_path = os.path.join(model_dir, "voice_encoder")
        if os.path.isdir(voice_encoder_path):
            try:
                ve = AutoModel.from_pretrained(voice_encoder_path)
                model.voice_encoder.load_state_dict(ve.state_dict(), strict=True)
                logging.info(
                    f"[TagSpeechBaseModel] Loaded voice_encoder from {voice_encoder_path}"
                )
            except Exception as e:
                logging.warning(
                    f"[TagSpeechBaseModel] Failed to load voice_encoder: {e}"
                )

        # Load LLM from subdirectory (if excluded from checkpoint)
        llm_path = os.path.join(model_dir, "llm")
        if os.path.isdir(llm_path):
            try:
                pretrained_llm = AutoModelForCausalLM.from_pretrained(llm_path)
                model.llm.load_state_dict(pretrained_llm.state_dict(), strict=False)
                logging.info(f"[TagSpeechBaseModel] Loaded LLM from {llm_path}")
            except Exception as e:
                logging.warning(f"[TagSpeechBaseModel] Failed to load LLM: {e}")

        model.eval()
        return model


class TagSpeechModel(TagSpeechBaseModel):
    """TagSpeech model with numeric time anchors for temporal alignment.

    This model inserts numeric anchors (1, 2, 3, ...) at regular intervals in both
    semantic and voice feature streams. The anchors use digit embeddings obtained
    through lookup tables, ensuring both branches insert the same numbered anchors
    at the same real-time positions to improve temporal alignment capability.

    Args:
        config: Model configuration
        tokenizer: Tokenizer instance
        digit_embeddings: Optional dict containing 'embeddings' (Tensor) and 'tokens' (List[str]).
                         If not provided, will be loaded in from_pretrained().
    """

    def __init__(self, config, tokenizer, digit_embeddings=None):
        super().__init__(config, tokenizer)

        if digit_embeddings is None:
            raise ValueError(
                "digit_embeddings must be provided. Pass a dict with 'embeddings' and 'tokens' keys."
            )

        # Validate input
        if not isinstance(digit_embeddings, dict):
            raise ValueError(
                f"digit_embeddings should be a dict containing 'embeddings' and 'tokens', "
                f"current type: {type(digit_embeddings)}"
            )

        embeddings = digit_embeddings.get("embeddings")
        tokens = digit_embeddings.get("tokens")

        if embeddings is None or tokens is None:
            raise ValueError(
                "digit_embeddings dict missing 'embeddings' or 'tokens' field"
            )

        embeddings = embeddings.float().contiguous()
        if embeddings.dim() != 2:
            raise ValueError(
                f"Digit embeddings shape should be [10, hidden], current: {embeddings.shape}"
            )

        self.register_buffer("_digit_embeddings", embeddings, persistent=False)
        self.digit_token_to_idx: Dict[str, int] = {
            tok: i for i, tok in enumerate(tokens)
        }

        missing = set("0123456789") - set(self.digit_token_to_idx.keys())
        if missing:
            raise ValueError(
                f"Digit embedding missing embeddings for the following characters: {sorted(missing)}"
            )

        if embeddings.shape[1] != self.llm.config.hidden_size:
            logging.warning(
                "[TagSpeechModel] digit embedding hidden size (%d) does not match LLM hidden size (%d), "
                "will adapt through runtime cast.",
                embeddings.shape[1],
                self.llm.config.hidden_size,
            )

        self._anchor_cache: Dict[int, torch.Tensor] = {}

        logging.info(
            "[TagSpeechModel] Loaded digit embeddings (dtype=%s, shape=%s). "
            "semantic_interval=%d, voice_interval=%d, insert_ends=%s",
            embeddings.dtype,
            tuple(embeddings.shape),
            config.semantic_anchor_interval,
            config.voice_anchor_interval,
            config.insert_anchors_at_ends,
        )

    @staticmethod
    def load_digit_embeddings(path: str) -> dict:
        """Load digit embeddings from file.

        Args:
            path: Path to digit embeddings file (.pt format)

        Returns:
            dict with 'embeddings' (Tensor) and 'tokens' (List[str])
        """
        digit_data = torch.load(path, map_location="cpu")
        if not isinstance(digit_data, dict):
            raise ValueError(
                f"Digit embedding file should be a dict containing 'embeddings' and 'tokens', "
                f"current type: {type(digit_data)}"
            )

        if "embeddings" not in digit_data or "tokens" not in digit_data:
            raise ValueError(
                f"Digit embedding file missing 'embeddings' or 'tokens' field"
            )

        return digit_data

    def _get_anchor_digits(self, anchor_idx: int) -> torch.Tensor:
        """Return cached digit embedding sequence for the given anchor index (CPU, float32)."""
        if anchor_idx < 1:
            anchor_idx = 1

        if anchor_idx not in self._anchor_cache:
            digits = [self.digit_token_to_idx[ch] for ch in str(anchor_idx)]
            index_tensor = torch.tensor(
                digits, dtype=torch.long, device=self._digit_embeddings.device
            )
            emb = torch.index_select(
                self._digit_embeddings, 0, index_tensor
            ).contiguous()
            self._anchor_cache[anchor_idx] = emb

        return self._anchor_cache[anchor_idx]

    def _insert_numeric_anchors(
        self,
        feats: torch.Tensor,
        lens: torch.Tensor,
        interval: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert numeric anchors into projector outputs."""
        B, T, D = feats.shape
        device = feats.device
        dtype = feats.dtype

        out_list: List[torch.Tensor] = []
        new_lens: List[int] = []
        max_len = 0

        for i in range(B):
            Li = int(lens[i].item())
            if Li <= 0:
                out_list.append(feats[i, :0])
                new_lens.append(0)
                continue

            seq = feats[i, :Li]

            insert_positions: List[int] = []
            if self.config.insert_anchors_at_ends:
                insert_positions.append(0)
            if interval > 0:
                pos = interval
                while pos < Li:
                    insert_positions.append(pos)
                    pos += interval
            if self.config.insert_anchors_at_ends and (
                len(insert_positions) == 0 or insert_positions[-1] != Li
            ):
                insert_positions.append(Li)

            chunks: List[torch.Tensor] = []
            prev = 0
            anchor_idx = 1

            for idx in insert_positions:
                idx = max(0, min(idx, Li))
                if idx > prev:
                    chunks.append(seq[prev:idx])

                anchor_cpu = self._get_anchor_digits(anchor_idx)
                anchor_emb = anchor_cpu.to(device=device, dtype=dtype)
                chunks.append(anchor_emb)

                anchor_idx += 1
                prev = idx

            if prev < Li:
                chunks.append(seq[prev:])

            if chunks:
                new_seq = torch.cat(chunks, dim=0)
            else:
                new_seq = seq

            out_list.append(new_seq)
            new_lens.append(new_seq.size(0))
            if new_seq.size(0) > max_len:
                max_len = new_seq.size(0)

        padded = []
        for seq in out_list:
            need = max_len - seq.size(0)
            if need > 0:
                pad = torch.zeros(need, D, device=device, dtype=dtype)
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)

        new_feats = torch.stack(padded, dim=0)
        new_lens_tensor = torch.tensor(new_lens, device=device, dtype=lens.dtype)
        return new_feats, new_lens_tensor

    def _forward_dual_audio_features(self, x: torch.Tensor, x_lens: torch.Tensor):
        semantic_features, voice_features, semantic_lens, voice_lens = (
            super()._forward_dual_audio_features(x, x_lens)
        )

        semantic_features, semantic_lens = self._insert_numeric_anchors(
            semantic_features, semantic_lens, self.config.semantic_anchor_interval
        )
        voice_features, voice_lens = self._insert_numeric_anchors(
            voice_features, voice_lens, self.config.voice_anchor_interval
        )

        return semantic_features, voice_features, semantic_lens, voice_lens
