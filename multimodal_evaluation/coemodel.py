import gc
from tqdm import tqdm
#from vllm import LLM, SamplingParams
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForSequenceClassification
from typing import List, Tuple, Dict, Union, Type, Optional
import lmms_eval.api.model
import lmms_eval.tasks
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.models.internvl2 import InternVL2
from lmms_eval.models.llava_onevision import Llava_OneVision
from lmms_eval.models.minicpm_v import MiniCPM_V

import lmms_eval
import pdb

max_model_length = 2048
max_new_tokens = 512


fine_class_list = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
                   'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
                   'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
                   'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
                   'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy',
                   'Physics', 'Psychology', 'Public_Health', 'Sociology']
fine_class_mapping = {i: fine_class for i, fine_class in enumerate(fine_class_list)}

model_lib = ['/data/vlm/coe/model/MiniCPM-V-2_6',
             '/data/vlm/coe/model/InternVL2-8B',
             '/data/vlm/coe/model/llava-onevision-qwen2-7b-ov']

#mmmu_test
subject_id_mapping={
            'Accounting' : 1,
            'Agriculture' : 2,
            'Architecture_and_Engineering' : 2,
            'Art' : 1, 
            'Art_Theory' : 2,
            'Basic_Medical_Science' : 1,
            'Biology' : 2,
            'Chemistry' : 1,
            'Clinical_Medicine' : 2,
            'Computer_Science' : 1,
            'Design' : 2,
            'Diagnostics_and_Laboratory_Medicine' : 1,
            'Economics' : 1, 
            'Electronics' : 1,
            'Energy_and_Power' : 2,
            'Finance' : 1,
            'Geography' : 2,
            'History' : 0,
            'Literature' : 1,
            'Manage' : 1,
            'Marketing' : 1,
            'Materials' : 1,
            'Math' : 1,
            'Mechanical_Engineering' : 0,
            'Music' : 0,
            'Pharmacy' : 1,
            'Physics' : 2,
            'Psychology' : 1,
            'Public_Health' : 1,
            'Sociology' : 1,
        }

@register_model("coemodel")
class COEModel(lmms):
    def __init__(
        self,
        pretrained: str = "model_lib",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_lib = model_lib
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cls_model_path = '/data/vlm/zxj/coe/bert_train/trained_model/bert_mmmu'
        self.cls_tokenizer = BertTokenizer.from_pretrained('/data/vlm/zxj/coe/bert_train/trained_model/bert_mmmu')
        self.cls_model = BertForSequenceClassification.from_pretrained(self.cls_model_path).to(self.device).eval()

        self.kwargs = kwargs

    def cls_question(self, question):
        inputs = self.cls_tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.cls_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        question_cls_id = torch.argmax(logits, dim=1).item()
        question_cls_class = fine_class_mapping[question_cls_id]

        return subject_id_mapping[question_cls_class]

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        pass

    def generate_until(self, requests) -> List[str]:
        
        print("len(requests):",len(requests))
        res = [None] * len(requests)
        classified_requests = {}
        classified_requests_ids ={}

        for i, request in enumerate(tqdm(requests)):
            #print("request:",request)

            question_text = request.arguments[0]
            model_index = self.cls_question(question_text)
            model_id = self.model_lib[model_index]
            
            if model_id not in classified_requests:
                classified_requests[model_id] = []
                classified_requests_ids[model_id] = []
            classified_requests[model_id].append(request)
            classified_requests_ids[model_id].append(i)

        for model_id, requests_batch in classified_requests.items():
            model = None
            if model_id == '/data/vlm/coe/model/MiniCPM-V-2_6':
                print(f"loading model: {model_id}", flush=True)
                model = MiniCPM_V(pretrained=model_id, task_dict=self.task_dict, **self.kwargs)
                responses = model.generate_until(requests_batch)
            elif model_id == '/data/vlm/coe/model/InternVL2-8B':
                print(f"loading model: {model_id}", flush=True)
                model = InternVL2(pretrained=model_id, task_dict=self.task_dict, **self.kwargs)
                responses = model.generate_until(requests_batch)
            elif model_id == '/data/vlm/coe/model/llava-onevision-qwen2-7b-ov':
                print(f"loading model: {model_id}", flush=True)
                model = Llava_OneVision(pretrained=model_id, task_dict=self.task_dict, **self.kwargs)
                responses = model.generate_until(requests_batch)
            
            for idx in range(len(responses)):
                original_index = classified_requests_ids[model_id][idx]
                res[original_index] = responses[idx]
        
            del model
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Unloaded model: {model_id}", flush=True)
        #print("res:",res)
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        pass
