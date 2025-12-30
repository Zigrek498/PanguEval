## 📌 Introduction
A comprehensive evaluation framework for **openPangu Model Series** across multiple domains.  
New models, benchmarks, and evaluation metrics can be easily customized.

---


## 🧪 Supported Benchmarks

| General Benchmarks  | Math Benchmarks | Code Benchmarks |
|---------------------|-----------------|-----------------|
| MMLU                | MATH-500        | LiveCodeBench   |
| MMLU-Pro            | AIME24          | MBPP+           |
| CMMLU               | AIME25          |                 |
| C-Eval              |                 |                 |
| GPQA-Diamond        |                 |                 |

---

## 🤖 Supported Models
### openPangu Series
* [openPangu-Embedded-1B-V1.1](https://huggingface.co/FreedomIntelligence/openPangu-Embedded-1B-V1.1)
* [openPangu-Embedded-7B-V1.1](https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B-V1.1)

---

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/Zigrek498/PanguEval
cd PanguEval

# Install dependencies
pip install -r requirements.txt
pip install 'open_clip_torch[training]'
pip install flash-attn --no-build-isolation

```

---

## 📂 Dataset Preparation
### HuggingFace Datasets
```python
MMLU: cais/mmlu
MMLU-Pro: TIGER-Lab/MMLU-Pro
CMMLU: haonan-li/cmmlu
C-Eval: ceval/ceval-exam
GPQA-Diamond: fingertap/GPQA-Diamond
MATH-500: HuggingFaceH4/MATH-500
AIME24: Maxwell-Jia/AIME_2024
AIME25: math-ai/aime25
LiveCodeBench: livecodebench/code_generation_lite
MBPP+: evalplus/mbppplus
```

---

## 🚀 Quick Start
### 1. Edit `eval_pangu.sh`
```bash
ASCEND_RT_VISIBLE_DEVICES="5"   # or "0,1,2,3"
MODEL_NAME="openPangu_1b"       # or "openPangu_7b"
MODEL_PATH="/opt/pangu/openPangu-Embedded-1B-V1.1"
THINKING_MODE="no_think"        # or "auto_think"/"think"
EVAL_DATASETS="CMMLU,MMLU_Pro,CEval,GPQA_Diamond,AIME24,AIME25"
```

### 2. Run Evaluation
```bash
bash eval_pangu.sh
```

### 3. Check Results
By default, the results are saved in `eval_results/${MODEL_NAME}_${DATETIME}`.

### 4. Customization
1. Upload new benchmarks in: `datas`
2. Model inference implementations in: `models`
3. Dataset evaluation utilities in: `utils`
4. Register new models via: `LLMs.py`
5. Register new datasets via: `benchmarks.py` & `utils/__init__.py`