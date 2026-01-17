import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


def forward(
    self,
    input_features,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    r"""
    Args:
        input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
            `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec libary (`pip install torchcodec`) or
            the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`torch.Tensor`)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """

    # expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    # if input_features.shape[-1] != expected_seq_length:
    #     raise ValueError(
    #         f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
    #     )

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    all_positions = torch.arange(
        inputs_embeds.shape[1], device=inputs_embeds.device
    )  # truncated to the exact length

    hidden_states = inputs_embeds + self.embed_positions(all_positions)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        if to_drop:
            layer_outputs = (None, None)
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)
    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, encoder_states, all_attentions] if v is not None
        )
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


def replace_whisper_encoder_forward():
    """
    This function monkey patches the forward method of the whisper encoder.
    To be called before the model is loaded, it changes whisper to process audio with any length < 30s.
    """
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    WhisperEncoder.forward = forward
