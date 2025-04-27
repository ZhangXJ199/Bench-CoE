import csv
import json
import argparse
import gc
import os
import pdb
from vllm import LLM, SamplingParams
import torch
import random
import transformers
import time
import re
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForSequenceClassification

from dataclasses import dataclass

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分类模型和tokenizer
cls_model_path = 'bert_query_mmlu_pro'
cls_tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
cls_model = BertForSequenceClassification.from_pretrained(cls_model_path).to(device).eval()


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

    return question_cls_id


class MyModel:
    def __init__(self, model_lib):
        self.model_lib = model_lib
        self.tokenizers = {}
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                              stop=["Question:"], top_k=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, model_id in enumerate(model_lib):
            if model_id in self.tokenizers:
                continue
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizers[model_id] = tokenizer

    def generate(self, test_df, val_df):
        response_batch = []
        model_questions = {}
        model_questions_ids = {}
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        logging.info(f"memory used1: {round(torch.cuda.memory_allocated() / 1024 / 1024 /1024)} GB")
        for i, curr in enumerate(test_df):
            k = args.ntrain
            curr = test_df[i]
            prompt_length_ok = False
            prompt = None
            model_index = cls_question(curr['question'])
            model_id = model_lib[model_index]
            tokenizer = self.tokenizers[model_id]
            while not prompt_length_ok:
                prompt = generate_cot_prompt(val_df, curr, k)
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                length = len(inputs["input_ids"][0])
                if length < max_model_length - max_new_tokens:
                    prompt_length_ok = True
                k -= 1

            if model_id not in model_questions:
                model_questions[model_id] = []
                model_questions_ids[model_id] = []
            model_questions[model_id].append(prompt)
            model_questions_ids[model_id].append(i)
        temp_response_dict = {}
        for model_id, prompt in model_questions.items():
            logging.info(f"loading LLM model: {model_id}")
            llm = LLM(model=model_id, gpu_memory_utilization=float(args.gpu_util),
                      tensor_parallel_size=torch.cuda.device_count(),
                      max_model_len=max_model_length,
                      trust_remote_code=True)
            logging.info(f"memory used2: {round(torch.cuda.memory_allocated() / 1024 / 1024 /1024)} GB")
            with torch.no_grad():
                outputs = llm.generate(prompt, self.sampling_params)
                for prompt_id, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    original_index = model_questions_ids[model_id][prompt_id]  # 获取对应的原始索引
                    temp_response_dict[original_index] = generated_text
            del llm
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            logging.info(f"Unloaded LLM model: {model_id}")
            logging.info(f"memory used3: {round(torch.cuda.memory_allocated() / 1024 / 1024 /1024)} GB")
        for i in range(len(test_df)):
            response_batch.append(temp_response_dict[i])

        del inputs, output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        gc.collect()
        logging.info(f"memory used4: {round(torch.cuda.memory_allocated() / 1024 / 1024 /1024)} GB")
        return response_batch

model_lib = ['../../../coe/model/gemma-2-9b-it',
             '../../../coe/model/Llama-3-Smaug-8B',
             '../../../coe/model/Mathstral-7B-v0.1',
             '../../../coe/model/Qwen2-7B-Instruct']

def load_mmlu_pro():
    dataset = load_dataset("../../../coe/data/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def load_model():
    llm = MyModel(model_lib)
    return llm

def preprocess(test_df):
    res_df = []
    for each in test_df:
        res_df.append(each)
    return res_df

def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]

def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"../cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def batch_inference(llm, test_df, val_df):
    start = time.time()
    logging.info("start a new subject")
    outputs = llm.generate(test_df, val_df)
    logging.info(str(len(test_df)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch

def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong

@torch.no_grad()
def eval_cot(subject, model, val_df, test_df, output_path):
    llm = model
    global choices
    logging.info("evaluating " + subject)

    pred_batch, response_batch = batch_inference(llm, test_df, val_df)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong

def main():
    logging.info(f"memory used0: {round(torch.cuda.memory_allocated() / 1024 / 1024 /1024)} GB")
    model = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count = eval_cot(subject, model, val_df, test_df, output_path)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="coe/coe_test")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()
