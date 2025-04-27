# name: CoE/eval_hellaswag_vllm_rule
# author：Mr.Zhao
# date：2024/10/27 
# time: 下午11:13 
# description:

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
    prompts = "Based on the context provided below, choose the index of the most plausible ending for the activity."
    return prompts


# 数据预处理函数
def preprocess(item):
    processed_data = []
    prompt = load_prompt()
    category_id, model_id = cls_question(item['ctx'])
    processed_data.append({
        'id': item['ind'],
        'input': item['ctx'],
        'options': item['endings'],
        'target': item['label'],
        'task': item['activity_label'],
        'prompt': prompt,
        'category_id': category_id,
        'model_id': model_id
    })
    return processed_data


def load_hellaswag_dataset(path='../coe/data/hellaswag/validation/validation.parquet'):
    dataset = []
    data = pd.read_parquet(path)
    for index, row in tqdm(data.iterrows()):
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
        # 加载每个LLM对应的tokenizer
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

                combined_input = f"{sample['prompt']}\nActivity: {sample['task']}\nContext: {sample['input']}\nOptions: \n1. {sample['options'][0]}\n2. {sample['options'][1]}\n3. {sample['options'][2]}\n4. {sample['options'][3]}\nAnswer: "

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
                      tensor_parallel_size=1,  # 单GPU
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
                [temp_response_dict[i], test_df.loc[i, 'options'], test_df.loc[i, 'target'], test_df.loc[i, 'task'],
                 test_df.loc[i, 'category_id'], test_df.loc[i, 'id']])

        return response_batch


def extract_answer(text):

    pattern = r"answer is: ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    pattern = r"answer is: ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None

def calculate_accuracy(response_batch):

    task_accuracy = {}
    category_accuracy = {}
    total_correct = 0
    total_count = 0

    for response in response_batch:
        generated_text, options, target, task, category, id = response

        extracted_answer = generated_text
        if extracted_answer!='':

            if '.' in extracted_answer:
                extracted_answer = extracted_answer.split('.')[0]
            if '\n' in extracted_answer:
                extracted_answer = extracted_answer.split('\n')[0]
            if ',' in extracted_answer:
                extracted_answer = extracted_answer.split(',')[0]
            if len(extracted_answer) > 1:
                extracted_answer = extracted_answer[0]
            if extracted_answer == '':
                is_correct = False
            else:
                is_correct = int(extracted_answer) == (int(target) + 1)
        else:
            is_correct = False

        if task not in task_accuracy:
            task_accuracy[task] = {'correct': 0, 'total': 0}
        if category not in category_accuracy:
            category_accuracy[category] = {'correct': 0, 'total': 0}

        task_accuracy[task]['total'] += 1
        category_accuracy[category]['total'] += 1

        if is_correct:
            task_accuracy[task]['correct'] += 1
            category_accuracy[category]['correct'] += 1

        total_count += 1
        if is_correct:
            total_correct += 1

    for task, stats in task_accuracy.items():
        task_accuracy[task]['accuracy'] = stats['correct'] / stats['total']

    for category, stats in category_accuracy.items():
        category_accuracy[category]['accuracy'] = stats['correct'] / stats['total']

    total_accuracy = total_correct / total_count

    return task_accuracy, category_accuracy, total_accuracy


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

# 主函数
def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'summary'))

    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    log_file_path = os.path.join(args.save_dir,f"summary/hellaswag_{time_str}_summary.txt")  # 例如，生成文件名如 "log_2024-10-28_14-30-00.txt"
    result_json_path = os.path.join(args.save_dir,f"hellaswag_{time_str}.json")  # JSON文件保存路径

    print(f"model loading start.", flush=True)
    model = MyModel(model_lib)


    print(f"dataset loading start.", flush=True)
    test_data = load_hellaswag_dataset()
    print(f"dataset loading finish.", flush=True)
    # print(test_data, flush=True)

    response = model.generate(test_data)

    task_accuracy, category_accuracy, total_accuracy = calculate_accuracy(response)

    examples_results = []
    for i, res in enumerate(response):
        example_result = {
            "id": str(res[5]),
            "input_sentence": test_data[i]['input'],
            "options": res[1].tolist(),
            "target_answer": str(res[2]),
            "category": str(res[3]),
            "model_output": str(res[0])
        }
        examples_results.append(example_result)

    save_example_results(examples_results, result_json_path)

    print("-------------------------------------------------")
    print(task_accuracy, flush=True)
    print("-------------------------------------------------")
    print(category_accuracy, flush=True)
    print("-------------------------------------------------")
    print("average_accuracy: ", total_accuracy, flush=True)

    with open(log_file_path, "w") as log_file:
        log_file.write("-------------------------------------------------\n")
        log_file.write(f"Task Accuracy: {task_accuracy}\n")
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

    cls_model_path = 'bert_query_hellaswag'
    cls_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cls_model = BertForSequenceClassification.from_pretrained(cls_model_path).to(device).eval()

    max_model_length = 4096
    max_new_tokens = 2048

    model_lib = model_pool[args.model]

    main()
