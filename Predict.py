import collections
import json

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer, GPT2Model

from LoraQuantizer import addLoraQuantizer
from evaluate import evaluate
from main import Gpt2ForExtractiveQA, test_collote_fn, batch_size, n_best, max_answer_length

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = 'gpt2'

dataset = load_dataset("squad")

test_data = dataset["validation"]
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token


config = AutoConfig.from_pretrained(checkpoint)
config.num_labels = 2
# F1: 1.33 EM: 0.00 AVG: 0.66
r = 1
w_bits = 16
a_bits = 16
model = Gpt2ForExtractiveQA.gpt2from_pretrained(checkpoint, config=config).to(device)
model = addLoraQuantizer(model, config, r=r, w_bits=w_bits, a_bits=a_bits).to(device)
loraPath = "epoch_1_valid_avg_4.6050_lora_weights.bin"
model.load_state_dict(torch.load(loraPath), strict=False)

# F1: 2.92 EM: 0.00 AVG: 1.46
# pretrained_model = GPT2Model.from_pretrained(checkpoint)
# model.gpt2.load_state_dict(pretrained_model.state_dict(), strict=True)

# F1: 47.02 EM: 63.68 AVG: 55.35
# model.load_state_dict(torch.load('epoch_3_valid_avg_48.6050_model_weights.bin'))

model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    all_example_ids = []
    all_offset_mapping = []
    for _, offset_mapping, example_ids in test_dataloader:
        all_example_ids += example_ids
        all_offset_mapping += offset_mapping
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    for batch_data, _, _ in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        pred_start_logits, pred_end_logit = model(batch_data)
        start_logits.append(pred_start_logits.cpu().numpy())
        end_logits.append(pred_end_logit.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)

    theoretical_answers = [
        {"id": test_data[s_idx]["id"], "answers": test_data[s_idx]["answers"]} for s_idx in range(len(test_dataloader))
    ]
    predicted_answers = []
    save_resluts = []
    for s_idx in tqdm(range(len(test_data))):
        example_id = test_data[s_idx]["id"]
        context = test_data[s_idx]["context"]
        title = test_data[s_idx]["title"]
        question = test_data[s_idx]["question"]
        labels = test_data[s_idx]["answers"]
        answers = []
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index - start_index + 1 > max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0],
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id,
                "prediction_text": best_answer["text"],
                "answer_start": best_answer["start"]
            })
            save_resluts.append({
                "id": example_id,
                "title": title,
                "context": context,
                "question": question,
                "answers": labels,
                "prediction_text": best_answer["text"],
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id,
                "prediction_text": "",
                "answer_start": 0
            })
            save_resluts.append({
                "id": example_id,
                "title": title,
                "context": context,
                "question": question,
                "answers": labels,
                "prediction_text": "",
                "answer_start": 0
            })
    eval_result = evaluate(predicted_answers, theoretical_answers)
    print(f"F1: {eval_result['f1']:>0.2f} EM: {eval_result['em']:>0.2f} AVG: {eval_result['avg']:>0.2f}\n")
    print('saving predicted results...')
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in save_resluts:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')