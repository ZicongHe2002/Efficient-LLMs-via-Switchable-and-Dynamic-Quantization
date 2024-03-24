# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention

from loralib.utils_quant import SymQuantizer


class GPT2QuantAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

        self.act_clip_val_k = torch.tensor([-2.0, 2.0])
        self.act_clip_val_v = torch.tensor([-2.0, 2.0])
        self.act_quantizer_k = SymQuantizer
        self.act_quantizer_v = SymQuantizer
        self.config = config

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

            if self.config.kv_bits < 32:
                key = self.act_quantizer_k.apply(
                    key, self.act_clip_val_k, self.kv_bits, False
                )
                value = self.act_quantizer_v.apply(
                    value, self.act_clip_val_v, self.kv_bits, False
                )

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2QuantBlock(GPT2Block):

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = GPT2QuantAttention(config, layer_idx=layer_idx)



class GPT2QuantModel(GPT2Model):
    def __init__(self, config, kv_bits=32):
        super().__init__(config)
        if not hasattr(config, 'kv_bits'):
            config.kv_bits = kv_bits
        self.h = nn.ModuleList([GPT2QuantBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.post_init()


