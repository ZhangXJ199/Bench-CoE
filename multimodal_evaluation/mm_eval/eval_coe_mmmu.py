import argparse
import torch
import os
import json
import random
import numpy as np
from tqdm import tqdm
import shortuuid

from TinyLLaVA_Factory.tinyllava.utils import *
from TinyLLaVA_Factory.tinyllava.data import *
from TinyLLaVA_Factory.tinyllava.model import *

from Bunny.bunny.model.builder import load_bunny_pretrained_model
from Bunny.bunny.util.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from Bunny.bunny.conversation import conv_templates
from Bunny.bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from PIL import Image
import math
from transformers import BertTokenizer, BertForSequenceClassification

bunny_subject = [0, 1, 3, 5, 6, 7, 9, 10, 12, 14, 17, 18, 19, 24, 27, 29]
tinyllava_subject = [2, 4, 8, 11, 13, 15, 16, 20, 21, 22, 23, 25, 26, 28]

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
        #pred_index = all_choices[0]
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def load_tinyllava_model(model_path):
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    return model, tokenizer, image_processor

def load_bunny_model(model_path, model_type):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_bunny_pretrained_model(model_path, None, model_name, model_type)
    return model, tokenizer, image_processor

def classify_question(question, bert_model, tokenizer):
    inputs = tokenizer.encode_plus(
        question,
        None,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    vocab_size = tokenizer.vocab_size
    input_ids = torch.where(input_ids >= vocab_size, torch.tensor(0), input_ids)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id


def eval_model(args):
    disable_torch_init()
    
    tinyllava_model, tinyllava_tokenizer, tinyllava_image_processor = load_tinyllava_model(args.model1_path)
    bunny4b_model, bunny4b_tokenizer, bunny4b_image_processor = load_bunny_model(args.model2_path, "phi-3")

    bert_model = BertForSequenceClassification.from_pretrained(args.bert_model_path)
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line["prompt"]

        predicted_class_id = classify_question(question, bert_model, bert_tokenizer)
        #print("predicted_class_id:",predicted_class_id)

        if predicted_class_id in tinyllava_subject:
            model_path = args.model1_path
            model = tinyllava_model
            tokenizer = tinyllava_tokenizer
            image_processor = tinyllava_image_processor

            text_processor = TextPreprocess(tokenizer, "phi")
            image_processor = ImagePreprocess(image_processor, model.config)
            model.to(device="cuda")

            if "image" in line:
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                image_sizes = [image.size]
                image = image_processor(image)
                images = image.unsqueeze(0).half().cuda()
                question = "<image>" + "\n" + question
            else:
                images = None
                image_sizes = None

            msg = Message()
            msg.add_message(question)

            result = text_processor(msg.messages, mode='eval')
            input_ids = result['input_ids']
            input_ids = input_ids.unsqueeze(0).cuda()

            with torch.inference_mode():
                if images is not None:
                    output_ids = model.generate(
                        input_ids,
                        images=images,
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=1024,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                else:
                    if line["question_type"] == "multiple-choice":
                        all_choices = line["all_choices"]
                        outputs = random.choice(all_choices)
                    else:
                        outputs = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"
        
        elif predicted_class_id in bunny_subject:
            model_path = args.model2_path
            model = bunny4b_model
            tokenizer = bunny4b_tokenizer
            image_processor = bunny4b_image_processor

            if "image" in line:
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                images = process_images([image], image_processor, model.config)[0]
            else:
                images = None

            def deal_with_prompt(input_text):
                qs = input_text
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                return qs

            prompt = deal_with_prompt(question)
            conv = conv_templates["bunny"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                if images is not None:
                    output_ids = model.generate(
                        input_ids,
                        images=images.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
                        #image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        top_p=None,
                        max_new_tokens=128,
                        use_cache=True,
                    )
                    input_token_len = input_ids.shape[1]
                    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                else:
                    if line["question_type"] == "multiple-choice":
                        all_choices = line["all_choices"]
                        outputs = random.choice(all_choices)
                    else:
                        outputs = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"
    
        if line["question_type"] == "multiple-choice":
            pred_ans = parse_multi_choice_response(
                outputs, line["all_choices"], line["index2ans"]
            )
        else:
            pred_ans = outputs

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": question,
                                   "text": pred_ans,
                                   "answer_id": ans_id,
                                   "model_id": model_path.split("/")[-1],
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_path", type=str, required=True)
    parser.add_argument("--model2_path", type=str, required=True)
    parser.add_argument("--bert_model_path", type=str, required=True)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    eval_model(args)
