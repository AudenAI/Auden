# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint as ckpt
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer import (
    ConformerEncoderLayer,
    TransformerEncoderLayer,
)
from wenet.utils.class_utils import (
    WENET_ACTIVATION_CLASSES,
    WENET_ATTENTION_CLASSES,
    WENET_EMB_CLASSES,
    WENET_MLP_CLASSES,
    WENET_NORM_CLASSES,
    WENET_SUBSAMPLE_CLASSES,
)
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import add_optional_chunk_mask, make_pad_mask


class BaseEncoder(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            query_bias: whether use bias in attention.linear_q
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            value_bias: whether use bias in attention.linear_v
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
            use_sdpa: whether to use SDPA, currently only support transformer for now
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        pos_emb_class = WENET_EMB_CLASSES[pos_enc_layer_type]
        # NOTE(Mddct): head_dim == output_size // attention_heads for most of
        #    speech tasks,  but for other task (LLM),
        #    head_dim == hidden_size * attention_heads. refactor later
        self.embed = WENET_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            (
                pos_emb_class(output_size, positional_dropout_rate)
                if pos_enc_layer_type != "rope_pos"
                else pos_emb_class(
                    output_size, output_size // attention_heads, positional_dropout_rate
                )
            ),
        )

        assert layer_norm_type in ["layer_norm", "rms_norm"]
        self.normalize_before = normalize_before
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size, eps=norm_eps)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        seq_packing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
            # Since we allow up to 1s(100 frames) delay, the maximum
            # chunk_size is 100 / 4 = 25.
            max_chunk_size=int(100.0 / self.embed.subsampling_rate),
        )
        if not seq_packing and self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
        if self.gradient_checkpointing and self.training:
            if seq_packing:
                xs = self.packed_forward_layers(
                    xs, chunk_masks, pos_emb, mask_pad, True
                )
            else:
                xs = self.forward_layers_checkpointed(
                    xs, chunk_masks, pos_emb, mask_pad
                )
        else:
            if seq_packing:
                xs = self.packed_forward_layers(
                    xs, chunk_masks, pos_emb, mask_pad, False
                )
            else:
                xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, chunk_masks

    def packed_forward_layers(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
        checkpointing: bool,
    ) -> torch.Tensor:
        from flash_attn.bert_padding import pad_input, unpad_input

        assert mask_pad.ndim == 3 and mask_pad.shape[1] == 1
        ori_batch_size = xs.shape[0]
        ori_seq_len = xs.shape[1]
        unpad_xs, indices, cu_seq_len, _ = unpad_input(xs, mask_pad[:, 0, :])
        unpad_xs, pad_size = self.pad_to_dividable(unpad_xs)
        unpad_xs = unpad_xs.unsqueeze(0)
        packed_attn_mask = self.construct_packed_attn_mask(
            chunk_masks, cu_seq_len, unpad_xs.shape[1]
        )

        # 构造kite attention需要的参数
        seq_len = unpad_xs.shape[1]
        num_head = self.encoders[0].self_attn.h
        head_dim = self.encoders[0].self_attn.d_k
        bs = 1

        cu_seqlens_q = torch.range(
            0, bs * seq_len, seq_len, dtype=torch.int32, device=packed_attn_mask.device
        )
        cu_seqlens_k = torch.range(
            0, bs * seq_len, seq_len, dtype=torch.int32, device=packed_attn_mask.device
        )
        max_seqlen_q = seq_len
        max_seqlen_k = seq_len
        m = None
        m_val = torch.zeros(bs * num_head, 1, seq_len, device=packed_attn_mask.device)

        kite_sparse_attn_mask = torch.zeros(
            bs, seq_len, 2, dtype=torch.int32, device=packed_attn_mask.device
        )
        kite_sparse_attn_mask[:, :, 1] = -1  # 所有的都标记为不可见

        # 找到第一个非0元素的下标
        # argmax(0) 返回每列第一个True的位置，如果没有非0元素则返回0
        non_zero_mask = packed_attn_mask[0]
        first_nonzero = torch.argmax(non_zero_mask.int(), dim=0)
        nonzero_len = non_zero_mask.sum(dim=0)

        kite_sparse_attn_mask[:, :, 0] = first_nonzero
        kite_sparse_attn_mask[:, :, 1] = first_nonzero + nonzero_len - 1

        from ptm_flash_attn.flash_attn_interface import convert_attn_mask_to_sparse_mask

        sparse_mask = convert_attn_mask_to_sparse_mask(
            attn_mask=kite_sparse_attn_mask,
            device=kite_sparse_attn_mask.device,
            head_dim=head_dim,
            is_dropout=False,
            is_causal=False,
            long_seq=True,
        )
        sparse_mask_bwd = convert_attn_mask_to_sparse_mask(
            attn_mask=kite_sparse_attn_mask,
            device=kite_sparse_attn_mask.device,
            head_dim=head_dim,
            is_dropout=False,
            is_causal=False,
            stage="bwd",
            long_seq=True,
        )
        m_u = torch.zeros(
            bs,
            num_head,
            seq_len,
            dtype=torch.int32,
            device=kite_sparse_attn_mask.device,
        )
        m_l = torch.zeros(
            bs,
            num_head,
            seq_len,
            dtype=torch.int32,
            device=kite_sparse_attn_mask.device,
        )
        m_u[:, :] = kite_sparse_attn_mask[0, :, 1]
        m_l[:, :] = kite_sparse_attn_mask[0, :, 0]

        kite_attn_args = (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            m,
            m_val,
            sparse_mask,
            sparse_mask_bwd,
            m_u,
            m_l,
        )
        # packed_attn_mask = mask_to_bias(packed_attn_mask, xs.dtype)

        for layer in self.encoders:
            if checkpointing:
                unpad_xs, packed_attn_mask, _, _ = ckpt.checkpoint(
                    layer.__call__,
                    unpad_xs,
                    packed_attn_mask,
                    pos_emb,
                    mask_pad,
                    kite_attn_args=kite_attn_args,
                    use_reentrant=False,
                )
            else:
                unpad_xs, packed_attn_mask, _, _ = layer(
                    unpad_xs,
                    packed_attn_mask,
                    pos_emb,
                    mask_pad,
                    kite_attn_args=kite_attn_args,
                )

        if pad_size > 0:
            unpad_xs = unpad_xs[:, :-pad_size]
        xs = pad_input(unpad_xs[0], indices, ori_batch_size, ori_seq_len)
        return xs

    def pad_to_dividable(self, xs, pad_length=128):
        assert xs.ndim == 2
        ori_len = xs.shape[0]
        import math

        target_length = math.ceil(ori_len / pad_length) * pad_length
        pad_len = target_length - xs.shape[0]
        xs = torch.nn.functional.pad(xs, (0, 0, 0, pad_len))
        return xs, pad_len

    def construct_packed_attn_mask(self, chunk_masks, cu_seq_len, max_seq_len):
        packed_len = cu_seq_len[-1]
        assert max_seq_len >= packed_len
        packed_attn_mask = torch.zeros(
            1, max_seq_len, max_seq_len, dtype=torch.bool, device=chunk_masks.device
        )
        for i in range(cu_seq_len.shape[0] - 1):
            # i is batch idx
            seq_start, seq_end = cu_seq_len[i], cu_seq_len[i + 1]
            seq_valid_len = seq_end - seq_start
            packed_attn_mask[0, seq_start:seq_end, seq_start:seq_end] = chunk_masks[
                i, :seq_valid_len, :seq_valid_len
            ]
        return packed_attn_mask

    def forward_layers(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    @torch.jit.unused
    def forward_layers_checkpointed(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = ckpt.checkpoint(
                layer.__call__, xs, chunk_masks, pos_emb, mask_pad, use_reentrant=False
            )
        return xs

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        decoding_chunk_size: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        assert decoding_chunk_size != -1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        offset = max(0, offset - 2)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        if offset == 0:
            start = 0
        else:
            start = 2
        xs = xs[:, start : start + decoding_chunk_size]
        pos_emb = pos_emb[:, start : start + decoding_chunk_size]
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(
            offset=offset - cache_t1, size=attention_key_size
        )
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            if elayers == 0:
                kv_cache = (att_cache, att_cache)
            else:
                i_kv_cache = att_cache[i : i + 1]
                size = att_cache.size(-1) // 2
                kv_cache = (i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :, size:])
            xs, _, new_kv_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=kv_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
            )
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            cur = max(0, cur - 4)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache, cnn_cache) = self.forward_chunk(
                chunk_xs,
                offset,
                required_cache_size,
                att_cache,
                cnn_cache,
                decoding_chunk_size=decoding_chunk_size,
            )
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        return ys, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        selfattention_layer_type: str = "selfattn",
        mlp_type: str = "position_wise_feed_forward",
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        """Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            static_chunk_size,
            use_dynamic_chunk,
            global_cmvn,
            use_dynamic_left_chunk,
            gradient_checkpointing,
            use_sdpa,
            layer_norm_type,
            norm_eps,
        )

        assert selfattention_layer_type in ["selfattn", "rope_abs_selfattn"]
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    output_size,
                    WENET_ATTENTION_CLASSES[selfattention_layer_type](
                        attention_heads,
                        output_size,
                        attention_dropout_rate,
                        query_bias,
                        key_bias,
                        value_bias,
                        use_sdpa,
                        n_kv_head,
                        head_dim,
                    ),
                    mlp_class(
                        output_size,
                        linear_units,
                        dropout_rate,
                        activation,
                        mlp_bias,
                        n_expert=n_expert,
                        n_expert_activated=n_expert_activated,
                    ),
                    dropout_rate,
                    normalize_before,
                    layer_norm_type=layer_norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(num_blocks)
            ]
        )


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = "position_wise_feed_forward",
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            static_chunk_size,
            use_dynamic_chunk,
            global_cmvn,
            use_dynamic_left_chunk,
            gradient_checkpointing,
            use_sdpa,
            layer_norm_type,
            norm_eps,
        )
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )
        # convolution module definition
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
            conv_bias,
        )

        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    WENET_ATTENTION_CLASSES[selfattention_layer_type](
                        *encoder_selfattn_layer_args
                    ),
                    mlp_class(*positionwise_layer_args),
                    mlp_class(*positionwise_layer_args) if macaron_style else None,
                    (
                        ConvolutionModule(*convolution_layer_args)
                        if use_cnn_module
                        else None
                    ),
                    dropout_rate,
                    normalize_before,
                    layer_norm_type=layer_norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(num_blocks)
            ]
        )
