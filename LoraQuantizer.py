# -*- coding: utf-8 -*-
import math

import torch
from torch import nn

from loralib import LoRALayer, Embedding, mark_only_lora_as_trainable, MergedLinear
from loralib.utils_quant import QuantizeLinear, SymQuantizer, AsymQuantizer


class LoraQuantizeLinear(QuantizeLinear, LoRALayer):
    # LoRA implemented in a dense layer
    # class LoRALayer(torch.nn.Module):
    #     def __init__(self, in_dim, out_dim, rank, alpha):
    #         super().__init__()
    #         std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
    #         self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
    #         self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
    #         self.alpha = alpha
    #
    #     def forward(self, x):
    #         x = self.alpha * (x @ self.A @ self.B)
    #          return x
    # lora_r: 8
    # lora_alpha: 1
    # lora_query: True
    # lora_key: False
    # lora_value: True
    # lora_projection: False
    # lora_mlp: True
    # lora_head: False
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 1, # 0->4
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        symmetric=True,
        bias=False,
        w_bits=32,
        a_bits=32,
        act_layerwise=False,
        weight_layerwise=False,
        **kwargs
    ):
        # device = kwargs.get("device", None)
        # dtype = kwargs.get("dtype", None)
        QuantizeLinear.__init__(self, in_features, out_features, symmetric=symmetric, w_bits=w_bits, a_bits=a_bits, act_layerwise=act_layerwise, weight_layerwise=weight_layerwise)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        # if mode:
        #     if self.merge_weights and self.merged:
        #         # Make sure that the weights are not merged
        #         if self.r > 0:
        #             self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = False
        # else:
        #     if self.merge_weights and not self.merged:
        #         # Merge the weights and mark it
        #         if self.r > 0:
        #             self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        # if self.r > 0 and not self.merged:
        #     # result = F.linear(x, T(self.weight), bias=self.bias)
        #     result = QuantizeLinear.forward(self, x)
        #     result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        #     return result
        # else:
        #     return QuantizeLinear.forward(self, x)
        result = QuantizeLinear.forward(self, x)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result


class QuantizeEmbedding(nn.Embedding, LoRALayer): # 可选参数
    # LoRA implemented in a dense layer
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 8,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            # symmetric=True,
            w_bits=32,
            a_bits=32,
            act_layerwise=False,
            weight_layerwise=False,
            **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise

        # if self.a_bits < 32 and self.a_bits > 2:
        #     if symmetric:
        #         self.act_quantizer = SymQuantizer
        #     else:
        #         self.act_quantizer = AsymQuantizer

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 32:
            weight = self.weight
        elif self.w_bits >= 3:
            weight_clip_val = torch.tensor([-2.0, 2.0])
            weight = SymQuantizer.apply(
                real_weights, weight_clip_val, self.w_bits, self.weight_layerwise
            )
        else:
            if self.w_bits == 1:
                if self.weight_layerwise:
                    scaling_factor = torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = torch.mean(
                        abs(real_weights), dim=1, keepdim=True
                    ).detach()
                quan_weights_no_grad = scaling_factor * (
                    torch.sign(real_weights / scaling_factor)
                )
            # elif self.w_bits == 2:
            #     scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            #     quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
            else:
                num_bits = 2 ** (self.w_bits - 1)
                clip_val = 1 - 1e-2
                if self.weight_layerwise:
                    scaling_factor = 2 * torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = (
                            2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
                    )
                quan_weights_no_grad = (
                        scaling_factor
                        * (
                                torch.round(
                                    torch.clamp(
                                        real_weights / scaling_factor, -clip_val, clip_val
                                    )
                                    * num_bits
                                    - 0.5
                                )
                                + 0.5
                        )
                        / num_bits
                )

            weight = (
                quan_weights_no_grad.detach() - real_weights.detach() + real_weights
            )
        # Quantize inputs
        # if self.a_bits < 32 and self.a_bits > 2:
        #     act_clip_val = torch.tensor([-2.0, 2.0])
        #     x = self.act_quantizer.apply(
        #         x, act_clip_val, self.a_bits, self.act_layerwise
        #     )

        result = nn.functional.embedding(
            x, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        if self.r > 0:
            after_A = nn.functional.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
        return result

        # if self.r > 0 and not self.merged:
        #     result = nn.Embedding.forward(self, x)
        #     after_A = F.embedding(
        #         x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
        #         self.norm_type, self.scale_grad_by_freq, self.sparse
        #     )
        #     result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
        #     return result
        # else:
        #     return nn.Embedding.forward(self, x)


def copyAToBWeight(a, b):
    if a.weight.shape == b.weight.shape:
        b.weight.data = a.weight.data.clone().detach()
    elif a.weight.shape == b.weight.shape[::-1]:
        b.weight.data = a.weight.data.clone().detach().transpose(0, 1)
    else:
        raise RuntimeError("Weight Shape ERROR")
    return b


EMBEDDING_KEY = "EMBEDDING"
ATTN_KEY = "ATTN"
MLP_KEY = "MLP"


def addLora(model, config):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    embed_dim = hidden_size
    max_position_embeddings = config.max_position_embeddings
    inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
    # wte vocab_size embed_dim
    # wpe max_position_embeddings embed_dim
    # GPT2Attention.c_attn hidden_size 3*hidden_size
    # GPT2Attention.c_proj hidden_size hidden_size
    # GPT2MLP.c_fc hidden_size inner_dim
    # GPT2MLP.c_proj inner_dim hidden_size
    model.gpt2.wte = copyAToBWeight(model.gpt2.wte, Embedding(vocab_size, embed_dim, r=8))
    model.gpt2.wpe = copyAToBWeight(model.gpt2.wpe, Embedding(max_position_embeddings, embed_dim, r=8))
    for layer in model.gpt2.h:
        layer.attn.c_attn = copyAToBWeight(layer.attn.c_attn, MergedLinear(hidden_size, 3 * hidden_size, r=8, enable_lora=[True, True, True]))
        layer.attn.c_proj = copyAToBWeight(layer.attn.c_proj, MergedLinear(hidden_size, hidden_size, r=8, enable_lora=[True, True, True]))
        layer.mlp.c_fc = copyAToBWeight(layer.mlp.c_fc, MergedLinear(hidden_size, inner_dim, r=8, enable_lora=[True, True, True]))
        layer.mlp.c_proj = copyAToBWeight(layer.mlp.c_proj, MergedLinear(inner_dim, hidden_size, r=8, enable_lora=[True, True, True]))
    model.train()
    mark_only_lora_as_trainable(model)
    return model

def addLoraQuantizer(model, config, r=8, w_bits=32, a_bits=32): #加参数，各层加参数想办法
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    embed_dim = hidden_size
    max_position_embeddings = config.max_position_embeddings
    inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
    # wte vocab_size embed_dim
    # wpe max_position_embeddings embed_dim
    # GPT2Attention.c_attn hidden_size 3*hidden_size
    # GPT2Attention.c_proj hidden_size hidden_size
    # GPT2MLP.c_fc hidden_size inner_dim
    # GPT2MLP.c_proj inner_dim hidden_size
    model.gpt2.wte = copyAToBWeight(model.gpt2.wte, QuantizeEmbedding(vocab_size, embed_dim, r=r, w_bits=w_bits))
    model.gpt2.wpe = copyAToBWeight(model.gpt2.wpe, QuantizeEmbedding(max_position_embeddings, embed_dim, r=r, w_bits=w_bits))
    for layer in model.gpt2.h:
        layer.attn.c_attn = copyAToBWeight(layer.attn.c_attn, LoraQuantizeLinear(hidden_size, 3 * hidden_size, r=r, w_bits=w_bits, a_bits=a_bits))
        layer.attn.c_proj = copyAToBWeight(layer.attn.c_proj, LoraQuantizeLinear(hidden_size, hidden_size, r=r, w_bits=w_bits, a_bits=a_bits))
        layer.mlp.c_fc = copyAToBWeight(layer.mlp.c_fc, LoraQuantizeLinear(hidden_size, inner_dim, r=r, w_bits=w_bits, a_bits=a_bits))
        layer.mlp.c_proj = copyAToBWeight(layer.mlp.c_proj, LoraQuantizeLinear(inner_dim, hidden_size, r=r, w_bits=w_bits, a_bits=a_bits))
    model.train()
    mark_only_lora_as_trainable(model)
    return model


def addLoraQuantizerWithConfig(model, config, lora_quant_config: dict):  # 加参数，各层加参数想办法
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    embed_dim = hidden_size
    max_position_embeddings = config.max_position_embeddings
    inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
    # wte vocab_size embed_dim
    # wpe max_position_embeddings embed_dim
    # GPT2Attention.c_attn hidden_size 3*hidden_size
    # GPT2Attention.c_proj hidden_size hidden_size
    # GPT2MLP.c_fc hidden_size inner_dim
    # GPT2MLP.c_proj inner_dim hidden_size
    model.gpt2.wte = copyAToBWeight(model.gpt2.wte, QuantizeEmbedding(vocab_size, embed_dim, **lora_quant_config.get(EMBEDDING_KEY)))
    model.gpt2.wpe = copyAToBWeight(model.gpt2.wpe, QuantizeEmbedding(max_position_embeddings, embed_dim, **lora_quant_config.get(EMBEDDING_KEY)))
    for layer in model.gpt2.h:
        layer.attn.c_attn = copyAToBWeight(layer.attn.c_attn, LoraQuantizeLinear(hidden_size, 3 * hidden_size, **lora_quant_config.get(ATTN_KEY)))
        layer.attn.c_proj = copyAToBWeight(layer.attn.c_proj, LoraQuantizeLinear(hidden_size, hidden_size, **lora_quant_config.get(ATTN_KEY)))
        layer.mlp.c_fc = copyAToBWeight(layer.mlp.c_fc, LoraQuantizeLinear(hidden_size, inner_dim, **lora_quant_config.get(MLP_KEY)))
        layer.mlp.c_proj = copyAToBWeight(layer.mlp.c_proj, LoraQuantizeLinear(inner_dim, hidden_size, **lora_quant_config.get(MLP_KEY)))
    model.train()
    mark_only_lora_as_trainable(model)
    return model


def modifyValues(model, config, r, kv_bits, w_bits, a_bits):
    modifyList = [model.gpt2.wte, model.gpt2.wpe]

    for layer in model.gpt2.h:
        modifyList.extend([layer.attn.c_attn, layer.attn.c_proj, layer.mlp.c_fc, layer.mlp.c_proj])

    for layer in modifyList:
        layer.r = r
        layer.w_bits = w_bits
        if not isinstance(layer, nn.Embedding):
            layer.a_bits = a_bits
    config.kv_bits = kv_bits
    return model


def modifyValuesWithConfig(model, lora_quant_config: dict, kv_bits=32):
    modifyEmbeddingList = [model.gpt2.wte, model.gpt2.wpe]
    modifyMLPList = []
    modifyATTNList = []
    modifyATTNKVList = []
    for layer in model.gpt2.h:
        modifyMLPList.extend([layer.mlp.c_fc, layer.mlp.c_proj])
        modifyATTNList.extend([layer.attn.c_attn, layer.attn.c_proj])
        modifyATTNKVList.extend([layer.attn, layer.attn])

    for layer in modifyEmbeddingList:
        layer.r = lora_quant_config.get(EMBEDDING_KEY)["r"]
        layer.w_bits = lora_quant_config.get(EMBEDDING_KEY)["w_bits"]
        # layer.a_bits = lora_quant_config.get(EMBEDDING_KEY)["a_bits"]
    for layer in modifyMLPList:
        layer.r = lora_quant_config.get(MLP_KEY)["r"]
        layer.w_bits = lora_quant_config.get(MLP_KEY)["w_bits"]
        layer.a_bits = lora_quant_config.get(MLP_KEY)["a_bits"]
    for layer in modifyATTNList:
        layer.r = lora_quant_config.get(ATTN_KEY)["r"]
        layer.w_bits = lora_quant_config.get(ATTN_KEY)["w_bits"]
        layer.a_bits = lora_quant_config.get(ATTN_KEY)["a_bits"]
    for layer in modifyATTNKVList:
        layer.config.kv_bits = kv_bits
    return model