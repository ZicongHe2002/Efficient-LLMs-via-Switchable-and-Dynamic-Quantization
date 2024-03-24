# -*- coding: utf-8 -*-
import torch
from transformers import AutoConfig, GPT2Model

from main import Gpt2ForExtractiveQA, addLora

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = "gpt2"
config = AutoConfig.from_pretrained(checkpoint)
config.num_labels = 2
model1 = Gpt2ForExtractiveQA.from_pretrained(checkpoint, config=config).to(device)

model2 = Gpt2ForExtractiveQA.gpt2from_pretrained(checkpoint, config=config).to(device)

model3 = Gpt2ForExtractiveQA(config=config).to(device)
pretrained_model = GPT2Model.from_pretrained(checkpoint)
model3.gpt2.load_state_dict(pretrained_model.state_dict(), strict=False)

# model = addLora(model3,config)
# 比较两个模型的参数
for (n1, p1), (n2, p2), (n3, p3) in zip(model1.gpt2.named_parameters(), model2.gpt2.named_parameters(), model3.gpt2.named_parameters()):
    # 检查参数名称是否一致
    assert n1 == n2, f"Parameter names mismatch: {n1} vs {n2}"
    assert n1 == n3, f"Parameter names mismatch: {n1} vs {n3}"


    # # 计算参数值的差异
    # diff = (p1 - p2).abs().max().item()
    #
    # # 打印参数名称和差异
    # print(f"Parameter: {n1}, Max difference: {diff}")
    # 计算参数值的差异
    diff = (p1 - p2).abs().sum().item()
    diff2 = (p1 - p3).abs().sum().item()
    diff3 = (p2 - p3).abs().sum().item()

    # 打印参数名称和差异
    print(f"Parameter: {n1}, Max difference: {diff} {diff2} {diff3}")




