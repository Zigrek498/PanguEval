import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm
import copy
import re

from ..utils import cal_acc
from ..base_dataset import BaseDataset

class CSarcasm(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.tasks = ["understanding", "classification", "response"]
        self.system_prompts = {}
        self.classification_map = { 
            "overstatement": "夸张",
            "understatement": "淡化",
            "contradiction": "反语",
            "fact": "事实",
            "metaphor": "比喻",
            "incongruity": "荒谬",
            "roleplay": "角色扮演"
        }
    
    def load_data(self):
        dataset = load_dataset(
            "json",
            data_files=f"{self.dataset_path}/test.json"
        )["train"]
        
        dataset = dataset.shuffle(seed=42)

        task_prompt_suffix = ""
        for task in self.tasks:
            prompt_path = f"{self.dataset_path}/prompt/{task}_cn{task_prompt_suffix}.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompts[task] = f.read()

        for task in self.tasks:
            for idx,sample in tqdm(enumerate(dataset)):
                if idx % self.num_chunks == self.chunk_idx:
                    sample = self.construct_messages(sample, task)
                    self.samples.append(sample)
        print(len(self.samples))
        return self.samples

    def construct_messages(self,original_sample,task):
        perm = ['A', 'B', 'C', 'D']
        sample = copy.deepcopy(original_sample)
        if task == self.tasks[0]:
            question = sample["question"]
            context = sample["comments"]
            option_A = sample[perm[0]]
            option_B = sample[perm[1]]
            option_C = sample[perm[2]]
            option_D = sample[perm[3]]
            answer = sample["Answer"]
            user_prompt = f"'context'：{context}\n'question'：{question}\n'A'：{option_A}\n'B'：{option_B}\n'C'：{option_C}\n'D'：{option_D}"
            
        elif task == self.tasks[1]:
            context = sample["comments"]
            answer = self.classification_map[sample["classification"]]
            user_prompt = f"'context'：{context}"
            
        else:
            context = sample["comments"]
            option_A = sample['roleplay'][perm[0]]
            option_B = sample['roleplay'][perm[1]]
            option_C = sample['roleplay'][perm[2]]
            option_D = sample['roleplay'][perm[3]]
            answer = sample['roleplay']["Answer"]
            user_prompt = f"'context'：{context}\n'A'：{option_A}\n'B'：{option_B}\n'C'：{option_C}\n'D'：{option_D}"

        messages = {"prompt":self.system_prompts[task]+"\n"+ user_prompt}
        sample["messages"] = messages
        sample["answer"] = answer
        sample["task"] = task
        return sample



    def cal_metrics(self, out_samples):
        metrics = {}
        for task in self.tasks:
            metrics[task] = {'total': 0, 'correct': 0}
        for item in out_samples:
            task = item['task']
            metrics[task]['total'] += 1
            resp = item['response'].split('\n\n')[-1].strip()
            if task in [self.tasks[0], self.tasks[2]]:
                model_choices = set(re.findall(r'[A-Da-d]', resp.upper()))
                correct_choices = set(item["answer"].upper())
                if model_choices == correct_choices:
                    metrics[task]['correct'] += 1
            elif task == self.tasks[1]:
                if item["answer"] in resp:
                    metrics[task]['correct'] += 1
        metrics = cal_acc(metrics)
        return metrics, out_samples

