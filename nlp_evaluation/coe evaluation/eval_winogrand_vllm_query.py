import os
import csv
import json
import argparse
import gc
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
import logging
import sys
import argparse

import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
from difflib import SequenceMatcher

model_pool = {
    'coe': ['../coe/model/gemma-2-9b-it',
            '../coe/model/Llama-3-Smaug-8B',
            '../coe/model/Mathstral-7B-v0.1',
            '../coe/model/Qwen2-7B-Instruct']
}

def load_prompt():
    prompts = "Based on the sentence below, select the most suitable word to complete the sentence meaningfully."
    return prompts

def preprocess(item):
    processed_data = []
    prompt = load_prompt()
    category_id, model_id = cls_question(item['sentence'])
    processed_data.append({
        'input': item['sentence'],
        'options': [item["option1"],item["option2"]],
        'target': item['answer'],
        'prompt': prompt,
        'category_id': category_id,
        'model_id': model_id,
        'id': item['qID']
    })
    return processed_data

def load_winogrande_dataset(path='../coe/data/winogrande_1.1/dev.jsonl'):
    dataset = []

    data = pd.read_json(path, lines=True)

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        example = preprocess(row)
        dataset.extend(example)
    return dataset


class MyModel:

    def __init__(self, model_lib):
        self.model_lib = model_lib
        self.tokenizers = {}
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                              stop=["Q:"], top_k=1)
        self.device = device

        for i, model_id in enumerate(model_lib):
            if model_id in self.tokenizers:
                continue
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizers[model_id] = tokenizer

    def generate(self, test_data):
        response_batch = []
        test_df = pd.DataFrame(test_data)
        grouped = test_df.groupby('model_id')

        model_questions = {}
        model_questions_ids = {}

        for model_id, group in grouped:
            logging.info(f"Loading LLM model: {model_id}")

            if model_id not in model_questions:
                model_questions[model_id] = []
                model_questions_ids[model_id] = []

            tokenizer = self.tokenizers[model_lib[model_id]]

            for index, sample in tqdm(group.iterrows()):

                combined_input = f"{sample['prompt']}\nSentence: {sample['input']}\nOptions: \n1. {sample['options'][0]}\n2. {sample['options'][1]}\nPlease choose the option (1 or 2) that best completes the sentence in a meaningful way. finish your answer with 'the answer is X' where X is the correct option's index."

                inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)
                while len(inputs["input_ids"][0]) >= max_model_length - max_new_tokens:
                    combined_input = combined_input[:-50]
                    inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)

                model_questions[model_id].append(combined_input)
                model_questions_ids[model_id].append(index)

        temp_response_dict = {}
        for model_id, prompts in model_questions.items():
            logging.info(f"Loading LLM model: {model_id}")
            llm = LLM(model=model_lib[model_id], gpu_memory_utilization=float(args.gpu_util),
                      tensor_parallel_size=1,
                      max_model_len=max_model_length,
                      trust_remote_code=True)
            logging.info(
                f"Memory used after loading model: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")

            with torch.no_grad():

                outputs = llm.generate(prompts, self.sampling_params)
                for prompt_id, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    original_index = model_questions_ids[model_id][prompt_id]
                    temp_response_dict[original_index] = generated_text

            del llm
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            logging.info(f"Unloaded LLM model: {model_id}")
            logging.info(
                f"Memory used after unloading model: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")

        for i in range(len(test_df)):
            response_batch.append(
                [temp_response_dict[i], test_df.loc[i, 'options'], test_df.loc[i, 'target'],
                 test_df.loc[i, 'category_id'], test_df.loc[i, 'id']])

        return response_batch

def extract_answer(text):
    pattern = r"answer is \(?(1|2)\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    pattern = r"answer is ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again_and_again(text)

def extract_again_and_again(text):
    pattern = r"answer is option ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    return text[0]


def calculate_accuracy(response_batch):
    category_accuracy = {}
    total_correct = 0
    total_count = 0

    for response in response_batch:
        generated_text, options, target, category, id = response
        print(generated_text, flush=True)
        extracted_answer = extract_answer(generated_text)

        if extracted_answer=='1' or extracted_answer=='2':
            is_correct = int(extracted_answer) == int(target)
        else:
            is_correct = False
        if category not in category_accuracy:
            category_accuracy[category] = {'correct': 0, 'total': 0}

        category_accuracy[category]['total'] += 1

        if is_correct:
            category_accuracy[category]['correct'] += 1

        total_count += 1
        if is_correct:
            total_correct += 1

    for category, stats in category_accuracy.items():
        category_accuracy[category]['accuracy'] = stats['correct'] / stats['total']

    total_accuracy = total_correct / total_count

    return category_accuracy, total_accuracy


def cls_question(question):
    inputs = cls_tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = cls_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    question_cls_id = logits.argmax().item()
    return question_cls_id, question_cls_id

def save_example_results(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        print(f"Example results saved to {file_path}")

def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    log_file_path = os.path.join(args.save_dir,
                                 f"winogrande_{time_str}_summary.txt")
    result_json_path = os.path.join(args.save_dir,f"winogrande_{time_str}.json")

    print(f"model loading start.", flush=True)
    model = MyModel(model_lib)

    print(f"dataset loading start.", flush=True)
    test_data = load_winogrande_dataset()
    print(f"dataset loading finish.", flush=True)

    response = model.generate(test_data)

    category_accuracy, total_accuracy = calculate_accuracy(response)

    examples_results = []
    for i, res in enumerate(response):
        extracted_answer = extract_answer(res[0])
        if extracted_answer=='1' or extracted_answer=='2':
            prediction = extracted_answer
        else:
            prediction = ""

        example_result = {
            "id": str(res[4]),
            "input_sentence": test_data[i]['input'],
            "options": res[1],
            "target_answer": str(res[2]),
            "category": str(res[3]),
            "model_output": str(res[0]),
            "prediction": prediction
        }
        examples_results.append(example_result)

    save_example_results(examples_results, result_json_path)

    print("-------------------------------------------------")
    print(category_accuracy, flush=True)
    print("-------------------------------------------------")
    print("average_accuracy: ", total_accuracy, flush=True)

    with open(log_file_path, "w") as log_file:
        log_file.write("-------------------------------------------------\n")
        log_file.write(f"Category Accuracy: {category_accuracy}\n")
        log_file.write("-------------------------------------------------\n")
        log_file.write(f"Average Accuracy: {total_accuracy}\n")
        log_file.write("-------------------------------------------------\n")


if __name__ == "__main__":
    global device, cls_model_path, cls_tokenizer, cls_model, max_model_length, max_new_tokens, model_lib

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="coe")
    parser.add_argument("--gpu_id", "-g", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    cls_model_path = 'bert_query_winogrande'
    cls_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cls_model = BertForSequenceClassification.from_pretrained(cls_model_path).to(device).eval()

    max_model_length = 4096
    max_new_tokens = 2048

    model_lib = model_pool[args.model]

    main()
