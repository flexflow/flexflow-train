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

OUTPUT_FOLDER="../benchmarking/data/req_rate_test"
TRACES_FOLDER="../benchmarking/traces"
MAX_SEQ_LEN=5000
BATCH_SIZE=8

NUM_BWD_LAYERS_PER_STEP=10

trace_files=(
    # sharegpt
    wildchat
)

arrival_rates=(
    0.20
    0.15
    0.10
    0.05
)
max_bwd_layers_per_step_values=(
    1
    2
    4
    16
)

max_tokens_per_batch_values=(
    128
    # 256
    # 512
)

mkdir -p $OUTPUT_FOLDER


# python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download(repo_id=\"${MODEL_NAME}\", allow_patterns=\"*.safetensors\", max_workers=30)"
# python ../inference/utils/download_hf_model.py $MODEL_NAME --half-precision-only
# python ../inference/utils/download_peft_model.py $PEFT_MODEL_NAME --half-precision-only
 
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_FILE=/usr/FlexFlow/inference/output/nccl2.log
# export LEGION_BACKTRACE=1
# export CUDA_VISIBLE_DEVICES=1,2,3,4

# Create trace files
for i in "${!trace_files[@]}"; do
for k in "${!arrival_rates[@]}"; do
    trace_file=${trace_files[$i]}
    arrival_rate=${arrival_rates[$k]}
    output_file="${TRACES_FOLDER}/${trace_files[$i]}_${arrival_rate}.json"
    # Create trace file if it does not exist
    if test -f $output_file; then
        echo "Trace file $output_file already exists"
        continue
    fi
    echo "Creating trace file $output_file"
    if [[ $trace_file == "sharegpt" ]]; then
        python ../benchmarking/get_sharegpt_trace.py  -o $output_file -t splitwise -a $arrival_rate
    elif [[ $trace_file == "wildchat" ]]; then
        python ../benchmarking/get_wildchat_trace.py -o $output_file -t splitwise -a $arrival_rate
    fi
done
done

for j in "${!max_tokens_per_batch_values[@]}"; do
for i in "${!trace_files[@]}"; do
for k in "${!arrival_rates[@]}"; do
    arrival_rate=${arrival_rates[$k]}
    TRACE_FILE="${TRACES_FOLDER}/${trace_files[$i]}_${arrival_rate}.json"
    test -f $TRACE_FILE || { echo "File $TRACE_FILE not found"; exit 1; }
    NUM_BWD_LAYERS_PER_STEP=${max_bwd_layers_per_step_values[$k]}
    
    MAX_TOKENS_PER_BATCH=${max_tokens_per_batch_values[$i]}

    LOG_FILE="${OUTPUT_FOLDER}/req_rate_${trace_files[$i]}_${MAX_TOKENS_PER_BATCH}_tokens_per_batch_${NUM_BWD_LAYERS_PER_STEP}_bwd_layers.log"
    rm $LOG_FILE || true
    
    echo "Running $TRACE_FILE with $MAX_TOKENS_PER_BATCH tokens/batch and $NUM_BWD_LAYERS_PER_STEP bwd layers/step"
    
    ./inference/peft/req_rate_benchmark \
        -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE \
        -llm-model $MODEL_NAME --fusion  \
        -tensor-parallelism-degree $NGPUS \
        -prompt $TRACE_FILE \
        -enable-peft -peft-model $PEFT_MODEL_NAME \
        --num-layers-per-finetuning-step $NUM_BWD_LAYERS_PER_STEP \
        -output-folder $OUTPUT_FOLDER \
        --max-requests-per-batch $BATCH_SIZE \
        --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
        --max-sequence-length $MAX_SEQ_LEN \
        2>&1 | tee $LOG_FILE
done
done
done
