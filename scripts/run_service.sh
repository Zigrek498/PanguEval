SERVED_MODEL_NAME="$1"
LOCAL_CKPT_DIR="$2"
PORT="$4"
export ASCEND_RT_VISIBLE_DEVICES="$3"
export VLLM_USE_V1=1
HOST=0.0.0.0

vllm serve $LOCAL_CKPT_DIR \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --max-num-seqs 16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 2048 \
    --tokenizer-mode "slow" \
    --dtype bfloat16 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill