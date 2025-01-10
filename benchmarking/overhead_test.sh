#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

reset
# ../config/config.linux
# make -j 
source ./set_python_envs.sh


MODEL_NAME="meta-llama/Llama-3.1-70B"
PEFT_MODEL_NAME="goliaro/llama3.1-70b-lora"
NGPUS=4
NCPUS=16
FSIZE=76000
ZSIZE=200000

# MODEL_NAME="JackFram/llama-160m"
# PEFT_MODEL_NAME="goliaro/llama-160m-lora"
# NGPUS=4
# NCPUS=16
# FSIZE=30000
# ZSIZE=20000

OUTPUT_FOLDER="../benchmarking/data/overhead_test"
MAX_SEQ_LEN=5000
BATCH_SIZE=8

max_tokens_per_batch_values=(
    128
    256
    512
)

max_finetuning_fwd_tokens_values=(
    "0,8,24,56,120"
    "0,8,24,56,120,248"
    "0,8,24,56,120,248,504"
)
max_finetuning_bwd_layers="0,1,2,4,8,16,32,64"

mkdir -p $OUTPUT_FOLDER


# python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download(repo_id=\"${MODEL_NAME}\", allow_patterns=\"*.safetensors\", max_workers=30)"
# python ../inference/utils/download_hf_model.py $MODEL_NAME --half-precision-only
# python ../inference/utils/download_peft_model.py $PEFT_MODEL_NAME --half-precision-only
 
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_FILE=/usr/FlexFlow/inference/output/nccl2.log
# export LEGION_BACKTRACE=1
# export CUDA_VISIBLE_DEVICES=1,2,3,4

for i in "${!max_tokens_per_batch_values[@]}"; do
    MAX_TOKENS_PER_BATCH=${max_tokens_per_batch_values[$i]}
    MAX_FINETUNING_FWD_TOKENS=${max_finetuning_fwd_tokens_values[$i]}
    LOG_FILE="${OUTPUT_FOLDER}/test_${MAX_TOKENS_PER_BATCH}_tokens_per_batch.log"
    rm $LOG_FILE || true
    ./inference/peft/overhead_test \
        -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE \
        -llm-model $MODEL_NAME --fusion  \
        -enable-peft -peft-model $PEFT_MODEL_NAME \
        -tensor-parallelism-degree $NGPUS \
        -output-folder $OUTPUT_FOLDER \
        --max-requests-per-batch $BATCH_SIZE \
        --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
        --max-sequence-length $MAX_SEQ_LEN \
        --max-fwd-finetuning-tokens $MAX_FINETUNING_FWD_TOKENS \
        --num-layers-per-finetuning-step $max_finetuning_bwd_layers \
        2>&1 | tee $LOG_FILE
done