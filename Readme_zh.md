## 📌 简介

一个面向 **openPangu 模型系列** 的综合评测框架，覆盖多个领域。
支持对新模型、新基准以及评测指标进行灵活定制。

---

## 🧪 支持的评测基准

| 通用评测基准       | 数学评测基准   | 代码评测基准        |
| ------------ | -------- | ------------- |
| MMLU         | MATH-500 | LiveCodeBench |
| MMLU-Pro     | AIME24   | MBPP+         |
| CMMLU        | AIME25   |               |
| C-Eval       |          |               |
| GPQA-Diamond |          |               |

---

## 🤖 支持的模型

### 开源盘古系列

* [openPangu-Embedded-1B-V1.1](https://huggingface.co/FreedomIntelligence/openPangu-Embedded-1B-V1.1)
* [openPangu-Embedded-7B-V1.1](https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B-V1.1)

---

## 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/Zigrek498/PanguEval
cd PanguEval

# 安装依赖
pip install -r requirements.txt
pip install 'open_clip_torch[training]'
pip install flash-attn --no-build-isolation
```

---

## 📂 数据集准备

### Hugging Face 数据集

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

## 🚀 快速开始

### 1. 编辑 `eval_pangu.sh`

```bash
ASCEND_RT_VISIBLE_DEVICES="5"   # 或 "0,1,2,3"
MODEL_NAME="openPangu_1b"       # 或 "openPangu_7b"
MODEL_PATH="/opt/pangu/openPangu-Embedded-1B-V1.1"
THINKING_MODE="no_think"        # 或 "auto_think"/"think"
EVAL_DATASETS="CMMLU,MMLU_Pro,CEval,GPQA_Diamond,AIME24,AIME25"
```

### 2. 运行评测

```bash
bash eval_pangu.sh
```

### 3. 查看结果

默认情况下，评测结果将保存在
`eval_results/${MODEL_NAME}_${DATETIME}` 目录下。

### 4. 自定义扩展

1. 上传新的评测基准：`datas`
2. 模型推理代码：`models`
3. 数据集评测代码：`utils`
4. 将新模型注册到评测框架中：`LLMs.py`
5. 将新数据集注册到评测框架中：`benchmarks.py` & `utils/__init__.py`