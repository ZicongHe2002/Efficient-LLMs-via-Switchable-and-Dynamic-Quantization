# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
from transformers import AutoConfig, GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from LoraQuantizer import addLoraQuantizer, addLora, addLoraQuantizerWithConfig
from LoraQuantizer import addLoraQuantizer, addLora, modifyValuesWithConfig, ATTN_KEY, EMBEDDING_KEY, MLP_KEY
from main import Gpt2ForExtractiveQA

def printBits(model):
    modifyEmbeddingList = [model.gpt2.wte, model.gpt2.wpe]
    modifyMLPList = []
    modifyATTNList = []
    modifyATTNKVList = []
    for layer in model.gpt2.h:
        modifyMLPList.extend([layer.mlp.c_fc, layer.mlp.c_proj])
        modifyATTNList.extend([layer.attn.c_attn, layer.attn.c_proj])
        modifyATTNKVList.extend([layer.attn, layer.attn])

    for layer in modifyEmbeddingList:
        print(f"{EMBEDDING_KEY}: r:{layer.r} w_bits:{layer.w_bits}")
    for layer in modifyMLPList:
        print(f"{MLP_KEY}: r:{layer.r} w_bits:{layer.w_bits} a_bits:{layer.a_bits}")
    for layer in modifyATTNList:
        print(f"{ATTN_KEY}: r:{layer.r} w_bits:{layer.w_bits} a_bits:{layer.a_bits}")
    for layer in modifyATTNKVList:
        print(f"kv_bits: {layer.config.kv_bits}")
    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = "gpt2"
config = AutoConfig.from_pretrained(checkpoint)
config.num_labels = 2
print(type(config))
print(hasattr(config, 'num_labels'))
print(hasattr(config, 'kv_bits'))

lora_quant_config = {
    EMBEDDING_KEY:{
        "r":8,
        "w_bits": 32,
        "a_bits" :32
    },
    ATTN_KEY:{
        "r":8,
        "w_bits": 32,
        "a_bits": 32,
        "lora_dropout":0.1
    },
    MLP_KEY:{
        "r": 8,
        "w_bits": 32,
        "a_bits": 32,
        "lora_dropout": 0.1
    }
}

model1 = Gpt2ForExtractiveQA.gpt2from_pretrained(checkpoint, config=config).to(device)

model2 = Gpt2ForExtractiveQA.gpt2from_pretrained(checkpoint, config=config).to(device)
model2 = addLoraQuantizerWithConfig(model2, config, lora_quant_config=lora_quant_config).to(device)

printBits(model2)
lora_quant_config[EMBEDDING_KEY]["w_bits"] = 16
lora_quant_config[ATTN_KEY]["w_bits"] = 16
lora_quant_config[MLP_KEY]["w_bits"]= 16
modifyValuesWithConfig(model2, lora_quant_config=lora_quant_config, kv_bits=32)# 动态修改bits
print("--------------------------------")
printBits(model2)
# 获取两个模型的 state_dict
state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()

# 查找新增的参数
print(model2)
added_params = {}

for name, param in state_dict1.items():
    print(f"Ori Parameter name: {name}, Shape: {param.shape}")
# for name, param in state_dict1.items():
#     print(f"Ori Parameter name: {name}, Shape: {param.shape}")
#
# for name, param in state_dict2.items():
#     print(f"Parameter name: {name}, Shape: {param.shape}")
#     if name not in state_dict1:
#         added_params[name] = param
#
# # 打印新增参数的名称和形状
# for name, param in added_params.items():
#     print(f"Delta Parameter name: {name}, Shape: {param.shape}")

for name, param in state_dict2.items():
    print(f"Parameter name: {name}, Shape: {param.shape}")
    if name not in state_dict1:
        added_params[name] = param

# 打印新增参数的名称和形状
for name, param in added_params.items():
    print(f"Delta Parameter name: {name}, Shape: {param.shape}")

