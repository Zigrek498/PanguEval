#!/bin/bash

# 获取用户输入
# read -p "请输入模型类型 (如 openPangu_1b): " MODEL_NAME
# read -p "请输入思考类型 (如 no_think): " THINKING_MODE
# read -p "请选择要评测的数据集 (逗号分隔，如 AIME24,AIME25): " EVAL_DATASETS

# 基本设置
ASCEND_RT_VISIBLE_DEVICES="5"
PORT=1041
MODEL_NAME="openPangu_1b"
MODEL_PATH="/opt/pangu/openPangu-Embedded-1B-V1.1"
THINKING_MODE="no_think"
# THINKING_MODE="auto_think"
EVAL_DATASETS="CMMLU,MMLU_Pro,CEval,GPQA_Diamond,AIME24,AIME25,LiveCodeBench,MBPP"

# 区分标准/非标准数据集
EXTRA_DATASETS="LiveCodeBench,MBPP"
EXTRA_SELECTED=""
NORMAL_SELECTED=""
IFS=',' read -ra EVAL_ARR <<< "$EVAL_DATASETS"
for ds in "${EVAL_ARR[@]}"; do
    if [[ ",$EXTRA_DATASETS," == *",$ds,"* ]]; then
        EXTRA_SELECTED+="${ds},"
    else
        NORMAL_SELECTED+="${ds},"
    fi
done
# 去掉末尾逗号
EXTRA_SELECTED="${EXTRA_SELECTED%,}"
NORMAL_SELECTED="${NORMAL_SELECTED%,}"


# 打印基本信息
echo "=============================="
echo "模型与硬件基本信息"
echo "=============================="
python scripts/print_base_info.py \
  --visible_devices ${ASCEND_RT_VISIBLE_DEVICES} \
  --model_path ${MODEL_PATH} \
  --log_name models.${MODEL_NAME}_base_info
sleep 5

# 设置其他参数
export HF_ENDPOINT=https://hf-mirror.com
DATASETS_PATH="datas"
OUTPUT_PATH="eval_results/$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="True"
SEED=42
REASONING="False"
TEST_TIMES=1
MAX_NEW_TOKENS=4096
MAX_IMAGE_NUM=6
TEMPERATURE=1
TOP_P=0.0001
REPETITION_PENALTY=1
USE_LLM_JUDGE="False"

# 打印配置信息
echo "=============================================="
echo "开始评测配置:"
echo "----------------------------------------------"
echo "模型名称: $MODEL_NAME"
echo "评测数据集: $EVAL_DATASETS"
echo "输出路径: $OUTPUT_PATH"
echo "随机种子: $SEED"
echo "最大新token数: $MAX_NEW_TOKENS"
echo "=============================================="

# 创建输出目录
mkdir -p "$OUTPUT_PATH"
OUTPUT_PATH=$(realpath "$OUTPUT_PATH")

# 运行标准数据集评测
if [[ -n "$NORMAL_SELECTED" ]]; then
    python eval.py \
        --eval_datasets "$NORMAL_SELECTED" \
        --datasets_path "$DATASETS_PATH" \
        --output_path "$OUTPUT_PATH" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --seed $SEED \
        --ascend_rt_visible_devices "$ASCEND_RT_VISIBLE_DEVICES" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --use_vllm "$USE_VLLM" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_image_num "$MAX_IMAGE_NUM" \
        --thinking_mode "$THINKING_MODE" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --reasoning "$REASONING" \
        --use_llm_judge "$USE_LLM_JUDGE" \
        --test_times "$TEST_TIMES"
fi

# 运行非标准数据集评测
if [[ -n "$EXTRA_SELECTED" ]]; then
    IFS=',' read -ra EXTRA_LIST <<< "$EXTRA_SELECTED"

    for ds in "${EXTRA_LIST[@]}"; do
        if [[ "$ds" == "LiveCodeBench" ]]; then
            pushd utils/LiveCodeBench >/dev/null || exit 1
    
            python -m lcb_runner.runner.main \
                --model "$MODEL_NAME" \
                --scenario codegeneration \
                --output_dir "$OUTPUT_PATH" \
                --local_model_path "$MODEL_PATH" \
                --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                --thinking_mode "$THINKING_MODE" \
                --temperature "$TEMPERATURE" \
                --top_p "$TOP_P" \
                --max_tokens "$MAX_NEW_TOKENS" \
                --trust_remote_code \
                --n 1 \
                --evaluate
    
            popd >/dev/null

        elif [[ "$ds" == "MBPP" ]]; then
        (
            bash scripts/run_service.sh "$MODEL_NAME" "$MODEL_PATH" "$ASCEND_RT_VISIBLE_DEVICES" "$PORT" &
            SERVICE_PID=$!
            sleep 300

            cleanup() {
                kill "$SERVICE_PID" 2>/dev/null
                wait "$SERVICE_PID" 2>/dev/null
            }
            trap cleanup EXIT
            
            pushd utils/MBPP/evalplus >/dev/null || exit 1
            mkdir -p "$OUTPUT_PATH/MBPP"

            python -m evalplus.evaluate \
                --model "$MODEL_NAME" \
                --dataset mbpp \
                --base-url http://localhost:$PORT/v1 \
                --backend openai \
                --output_file "$OUTPUT_PATH/MBPP/eval_results.json" \
                --temperature "$TEMPERATURE" \
                --tp "$TENSOR_PARALLEL_SIZE" \
                --n_samples 2 \
                --greedy
    
            popd >/dev/null
        )
        fi
    done
fi


echo "=============================================="
echo "评测完成! 结果保存在: $OUTPUT_PATH"
echo "=============================================="