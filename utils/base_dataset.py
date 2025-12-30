import torch
import os
import json

from tqdm import tqdm
import gc

from .utils import save_json

class BaseDataset:
  def __init__(self):
    self.chunk_idx = int(os.environ.get("chunk_idx",0))
    self.num_chunks = int(os.environ.get("num_chunks",1))


  def run(self,samples,model):
    out_samples = []
    with torch.no_grad():
        for sample in tqdm(samples):
            messages = sample["messages"]
            if len(messages) > 2:
                output = model.generate_output_multi_turn(messages)
            elif len(messages) > 1 and messages[0]['role'] == 'user':
                output = model.generate_output_multi_turn(messages)
            else:
                output = model.generate_output(messages)
            try:
                if len(messages) == 1:
                  del sample["messages"]
                sample["response"] = output
                out_samples.append(sample)
            except Exception as e:
                from pdb import set_trace;set_trace()
                print(e)
            gc.collect()
    return out_samples

  def cal_matrics(self):
    pass

  def init_dataset(self):
    pass

  def construct_messages(self):
    pass


  def eval(self):
      model = self.model
      dataset_path = self.dataset_path
      output_path = self.output_path
      num_chunks = self.num_chunks
      chunk_idx = self.chunk_idx
      if num_chunks == 1:
          results_path = os.path.join(output_path,"results.json")
          matric_path = os.path.join(output_path,"metrics.json")
          if not os.path.exists(results_path):
            out_samples = self.run(self.samples,model)
            save_json(results_path,out_samples)
          else: # 已有推理结果，且大小相同，才跳过推理
            with open(results_path, 'r') as f:
              records = json.load(f)
            if len(records) == len(self.samples):
              out_samples = records
            else:
              out_samples = self.run(self.samples,model)
              save_json(results_path,out_samples)

          metrics,out_samples = self.cal_metrics(out_samples)
          save_json(matric_path,metrics)
          save_json(results_path,out_samples)
          return metrics


      elif num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        final_results_path = os.path.join(output_path,"results.json")
        out_samples = self.run(self.samples,model)
        save_json(results_path,out_samples)

        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            total_results = []
            for result in total_results_path:
                results_path = os.path.join(output_path,result)
                with open(results_path,"r") as f:
                    total_results.extend(json.load(f))

            save_json(final_results_path,total_results)
            metrics,out_samples = self.cal_metrics(total_results)
            matric_path = os.path.join(output_path,"metrics.json")
            save_json(matric_path,metrics)
            save_json(final_results_path,out_samples)
            return metrics
        else:
            return None
      else:
          raise ValueError("num_chunks must be greater than 0")