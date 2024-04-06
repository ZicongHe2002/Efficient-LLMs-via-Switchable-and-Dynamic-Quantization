# -*- coding: utf-8 -*-
import collections
import os
import random
import sys
from math import inf
from typing import Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig
from transformers import GPT2PreTrainedModel
from transformers import get_scheduler
from transformers import AdamW, get_linear_schedule_with_warmup
from LoraQuantizer import EMBEDDING_KEY, ATTN_KEY, MLP_KEY, addLoraQuantizerWithConfig, modifyValuesWithConfig
from evaluate2 import evaluate
from gpt2_quant import GPT2QuantModel
from loralib import mark_only_lora_as_trainable, lora_state_dict

sys.path.append('./')

max_length = 384
stride = 128
n_best = 20
max_answer_length = 30
batch_size = 8
learning_rate = 5e-5
weight_decay = 0.01
epoch_num = 1000
r = 1
# w_bits = 16
a_bits = 32
w_bits = [8,16,32]
loraPath = ""
#  新加的 step 5
current_w_bits = 32
lora_quant_configs = {
        EMBEDDING_KEY: {"r": 8, "w_bits": 32, "a_bits": 32},
        ATTN_KEY: {
            "layer_0_3": {"r":8, "w_bits": 32, "a_bits": 32, "lora_dropout": 0.1},
            "layer_4_7": {"r": 8, "w_bits": 16, "a_bits": 32, "lora_dropout": 0.1},
            "layer_8_11": {"r": 8, "w_bits": 8, "a_bits": 32, "lora_dropout": 0.1}
        },
        MLP_KEY: {
            "layer_0_3": {"r": 8, "w_bits": 32, "a_bits": 32, "lora_dropout": 0.1},
            "layer_4_7": {"r": 8, "w_bits": 16, "a_bits": 32, "lora_dropout": 0.1},
            "layer_8_11": {"r": 8, "w_bits": 8, "a_bits": 32, "lora_dropout": 0.1}
        }
    }
train_set_size = 78839
valid_set_size = 8760
checkpoint = 'gpt2'

# main (1).py

# 在文件顶部定义LayerConfig类
# 确保其他导入和配置代码后面添加这个

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token


def train_collote_fn(batch_samples):
    batch_question, batch_context, batch_answers = [], [], []
    for sample in batch_samples:
        batch_question.append(sample['question'].strip())
        batch_context.append(sample['context'].strip())
        batch_answers.append(sample['answers'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        return_tensors="pt"
    )

    offset_mapping = batch_data.pop('offset_mapping')
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = batch_answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = batch_data.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    return batch_data, torch.tensor(start_positions), torch.tensor(end_positions)


def test_collote_fn(batch_samples):
    batch_id, batch_question, batch_context = [], [], []
    for sample in batch_samples:
        batch_id.append(sample['id'])
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    offset_mapping = batch_data.pop('offset_mapping').numpy().tolist()
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(batch_data['input_ids'])):
        sample_idx = sample_mapping[i]
        example_ids.append(batch_id[sample_idx])

        sequence_ids = batch_data.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    return batch_data, offset_mapping, example_ids


class Gpt2ForExtractiveQA(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt2 = GPT2QuantModel(config)
        self.lora_start_fc = nn.Linear(config.vocab_size, 1)
        self.lora_end_fc = nn.Linear(config.vocab_size, 1)

        self.post_init()

    def forward(self, X, label=None):
        input_ids = X['input_ids'].to(device)
        attention_mask = X['attention_mask'].to(device)
        # 第一次GPT-2输出
        gpt2_output = self.gpt2(input_ids, attention_mask=attention_mask, token_type_ids=None)
        start_logits = self.lora_start_fc(gpt2_output.logits).squeeze(-1).contiguous()

        if label is None:
            label = start_logits.argmax(dim=-1)
        label = label.unsqueeze(-1).contiguous()

        input_ids = torch.cat((input_ids, label), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones(input_ids.size(0), 1).to(input_ids.device)), dim=-1)

        gpt2_output = self.gpt2(input_ids, attention_mask=attention_mask, token_type_ids=None)
        end_logits = self.lora_end_fc(gpt2_output.logits).squeeze(-1).contiguous()

        return start_logits, end_logits

    @classmethod
    def gpt2from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            use_safetensors: bool = None,
            **kwargs,
    ):
        instance = cls(config=config)
        instance.gpt2 = instance.gpt2.from_pretrained(pretrained_model_name_or_path, *model_args, config=config,
                                                      cache_dir=cache_dir,
                                                      ignore_mismatched_sizes=ignore_mismatched_sizes,
                                                      force_download=force_download,
                                                      local_files_only=local_files_only, token=token, revision=revision,
                                                      use_safetensors=use_safetensors, **kwargs)
        return instance


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)
    lora_quant_configs = {
        EMBEDDING_KEY: {"r": 8, "w_bits": 32, "a_bits": 32},
        ATTN_KEY: {
            "layer_0_3": {"r":8, "w_bits": 32, "a_bits": 32, "lora_dropout": 0.1},
            "layer_4_7": {"r": 8, "w_bits": 16, "a_bits": 32, "lora_dropout": 0.1},
            "layer_8_11": {"r": 8, "w_bits": 8, "a_bits": 32, "lora_dropout": 0.1}
        },
        MLP_KEY: {
            "layer_0_3": {"r": 8, "w_bits": 32, "a_bits": 32, "lora_dropout": 0.1},
            "layer_4_7": {"r": 8, "w_bits": 16, "a_bits": 32, "lora_dropout": 0.1},
            "layer_8_11": {"r": 8, "w_bits": 8, "a_bits": 32, "lora_dropout": 0.1}
        }
    }
    model.train()
    mark_only_lora_as_trainable(model)
    modifyValuesWithConfig(model, lora_quant_configs)
    for batch, (X, start_pos, end_pos) in enumerate(dataloader, start=1):
        X, start_pos, end_pos = X.to(device), start_pos.to(device), end_pos.to(device)

        start_pred, end_pred = model(X, label=start_pos)
        start_loss = loss_fn(start_pred, start_pos)
        end_loss = loss_fn(end_pred, end_pos)
        loss = (start_loss + end_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, dataset, model, tokenizer):
    all_example_ids = []
    all_offset_mapping = []
    for _, offset_mapping, example_ids in dataloader:
        all_example_ids += example_ids
        all_offset_mapping += offset_mapping
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    for batch_data, _, _ in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            pred_start_logits, pred_end_logit = model(batch_data)
        start_logits.append(pred_start_logits.cpu().numpy())
        end_logits.append(pred_end_logit.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)

    predicted_answers = []
    for example_id, feature_indices in example_to_features.items():
        example = next((dataset[idx] for idx in range(len(dataset)) if dataset[idx]['id'] == example_id), None)
        if example is None:
            print(f"Error: Could not find example with id {example_id}")
            continue
        context = example["context"]

        min_null_score = None
        valid_answers = []

        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        valid_answers.append(("", 0))  # 添加一个空答案,分数为0
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        valid_answers.append(("", 0))  # 添加一个空答案,分数为0
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    try:
                        answer = context[offsets[start_index][0]: offsets[end_index][1]]
                        answer = tokenizer.decode(tokenizer.encode(answer))
                    except IndexError:
                        valid_answers.append(("", 0))  # 添加一个空答案,分数为0
                        continue

                    score = start_logit[start_index] + end_logit[end_index]
                    if not answer:
                        if min_null_score is None or score < min_null_score:
                            min_null_score = score
                    else:
                        valid_answers.append((answer, score))
        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x[1])
            predicted_answers.append({
                "id": example_id,
                "prediction_text": best_answer[0]
            })
        else:
            predicted_answers.append({
                "id": example_id,
                "prediction_text": ""
            })

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
    print(f"Number of predictions: {len(predicted_answers)}")
    print(f"Number of theoretical answers: {len(theoretical_answers)}")
    result = evaluate(predicted_answers, theoretical_answers)
    print(f"BLEU: {result['bleu']:>0.4f} ROUGE-L: {result['rouge']:>0.4f} METEOR: {result['meteor']:>0.4f}\n")
    return result

if __name__ == '__main__':
    dataset = load_dataset("squad")
    train_data, valid_data = random_split(dataset["train"], [train_set_size, valid_set_size])
    test_data = dataset["validation"]

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_collote_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)

    print('train set size: ', )
    print(len(train_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in train_dataloader]))
    print('valid set size: ')
    print(len(valid_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in valid_dataloader]))
    print('test set size: ')
    print(len(test_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in test_dataloader]))

    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = 2
    model = Gpt2ForExtractiveQA.gpt2from_pretrained(checkpoint, config=config).to(device)
   # model = addLoraQuantizer(model, config, r=r, w_bits=w_bits, a_bits=a_bits).to(device)
    model = addLoraQuantizerWithConfig(model, config, lora_quant_config=lora_quant_configs).to(device)
    #之前训练的lora，可以保存的路径，就是继承之前的训练数据。
    if loraPath:
        model.load_state_dict(torch.load(loraPath), strict=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epoch_num * len(train_dataloader)
    warmup_steps = int(total_steps * 0.1)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    total_loss = 0.
    best_avg_score = 0.
    best_loss = inf
    for t in range(epoch_num): #想办法保存epoch的次数，然后这里从epoch的次数开始训练，
        if t <= 1:
            current_w_bits = w_bits[0]
            print(f"Epoch {t} with w_bits {current_w_bits}_learnrate_{learning_rate}_learnsche_{lr_scheduler}\n-------------------------------")
        elif t <= 10:
            current_w_bits = w_bits[1]
            print(f"Epoch {t} with w_bits {current_w_bits}\n-------------------------------")
        else:
            current_w_bits = w_bits[2]
            print(f"Epoch {t} with w_bits {current_w_bits}\n-------------------------------")
        modifyValuesWithConfig(model, lora_quant_config=lora_quant_configs, kv_bits=32)# 动态修改bits

        print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss)
        generated_answers = test_loop(valid_dataloader, valid_data, model, tokenizer)
        valid_scores = test_loop(valid_dataloader, valid_data, model, tokenizer)
        avg_score = valid_scores['avg']
        if total_loss < best_loss:
            best_loss = total_loss
        # if avg_score > best_avg_score:
        #     best_avg_score = avg_score
            print('saving new weights...\n')
            lora_dict = lora_state_dict(model)
            if len(lora_dict) == 0:
                torch.save(model.state_dict(), f'epochgdtg_{t + 1}_valid_avg_{avg_score:0.4f}_model_weights.bin')
            else:
                torch.save(lora_dict, f'epoch_{t + 1}_valid_avg_{avg_score:0.4f}_lora_weights.bin')
    print("Done!")