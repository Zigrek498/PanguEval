import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Load OpenPangu model on Ascend NPU")
    parser.add_argument(
        "--visible_devices",
        type=str,
        default="0",
        help="ASCEND_RT_VISIBLE_DEVICES, e.g. '0' or '0,1'"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="models.openPangu_1b_vllm",
        help="Logger name"
    )
    return parser.parse_args()

args = parse_args()

import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices

import logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(args.log_name)
logger.setLevel(logging.INFO)

import torch
npu_num = torch.npu.device_count()
logger.info(f"检测到 {npu_num} 个NPU设备")

for i in range(npu_num):
    prop = torch.npu.get_device_properties(i)
    logger.info(f"NPU {i}: {prop.name}, 总显存: {prop.total_memory/(1024**3):.2f} GB")

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("npu")
print(model.device, model)