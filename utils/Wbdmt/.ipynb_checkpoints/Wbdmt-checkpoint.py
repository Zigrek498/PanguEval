import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm
import re

from ..utils import cal_acc
from ..base_dataset import BaseDataset

class Wbdmt(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.cates = ["步数最优化", "路径计数", "一笔画连线", "坐标点连线"]

    def load_data(self):
        dataset = load_dataset(
            "json",
            data_files=f"{self.dataset_path}/test.jsonl"
            # data_files=f"{self.dataset_path}/wbdmt_{self.cates[4]}.jsonl"
        )["train"]
        
        dataset = dataset.shuffle(seed=42).select(range(2))
        
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        answer = sample["answer"]
        question = sample["question"]

        messages = {"prompt":question}
        sample["prompt"] = question
        sample["messages"] = messages
        sample["answer"] = answer
        return sample



    def cal_metrics(self, out_samples):
        metrics = {}
        for cate in self.cates:
            metrics[cate] = {'total': 0, 'correct': 0}
        metrics['total'] = 0
        metrics['correct'] = 0
        for item in out_samples:
            cate = item['category']
            metrics[cate]['total'] += 1
            metrics['total'] += 1
            resp = item['response'].split('\n\n')[-1].strip()
            answer = item['answer'].strip()
            if cate in ["步数最优化", "路径计数"]:
                pattern = r'(?<!\d)\d+(?!\d)'
                matches = re.findall(pattern, resp)
                if len(matches) == 1 and matches[0] == answer:
                    metrics[cate]['correct'] += 1
                    metrics['correct'] += 1
            elif cate == "一笔画连线":
                seq = [a.strip() for a in answer.split('-')]
                order_pattern = ".*".join(seq)
                unique_letters = set(seq)
                once_constraints = "".join(
                    rf"(?!.*{ch}.*{ch})" for ch in unique_letters
                )
                pattern = re.compile(rf"^{once_constraints}{order_pattern}$")
                if pattern.search(resp):
                    metrics[cate]['correct'] += 1
                    metrics['correct'] += 1
            elif cate == "坐标点连线":
                ans = answer.split('\n')
                seq = []
                for a in ans:
                    seq.append(a[0])
                    seq.append(a[-1])
                order_pattern = ".*".join(seq)
                unique_points = set(seq)
                once_constraints = "".join(
                    rf"(?!.*{re.escape(ch)}.*{re.escape(ch)})" for ch in unique_points
                )
                pattern = re.compile(rf"^{once_constraints}{order_pattern}$", re.DOTALL)
                if pattern.search(resp):
                    metrics[cate]['correct'] += 1
                    metrics['correct'] += 1
            else:
                pass
        metrics = cal_acc(metrics)
    
        return metrics, out_samples
