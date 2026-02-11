import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm
from collections import Counter

from ..utils import safe_load_json_response, cal_acc
from ..base_dataset import BaseDataset

class Wbsxcq(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        dataset = load_dataset(
            "json",
            data_files=f"{self.dataset_path}/*.json"
        )["train"]
        
        dataset = dataset.shuffle(seed=42)
        
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        answer = sample["task"]["answer"]
        prompt = "Answer the question based on the text. Output only a valid JSON array of strings and nothing else."
        user_content = (
            f"Text:\n{sample['text']}\n\n"
            f"Question:\n{sample['task']['question']}\n\n"
            f"{prompt}"
        )
        messages = {"prompt":user_content}
        sample["prompt"] = user_content
        sample["messages"] = messages
        sample["answer"] = answer
        return sample



    def cal_metrics(self, out_samples):
        metrics = {}
        metrics['total'] = 0
        metrics['correct'] = 0
    
        for i, sample in enumerate(out_samples):
            metrics['total'] += 1
            resp = sample["response"].strip()
            answer = sample["answer"]
    
            try:
                resp = safe_load_response(resp)
            except:
                resp = None
            if Counter(answer) == Counter(resp):
                metrics['correct'] += 1
        metrics = cal_acc(metrics)
    
        return metrics, out_samples

