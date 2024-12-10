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
NGPUS=4
NCPUS=16
FSIZE=76000
ZSIZE=200000

OUTPUT_FOLDER="../benchmarking/data/incr_decoding"
TRACES_FOLDER="../benchmarking/traces"
MAX_SEQ_LEN=5000
BATCH_SIZE=8

trace_files=(
    sharegpt
    wildchat
)

max_tokens_per_batch_values=(
    512
    256
    128
)

mkdir -p $OUTPUT_FOLDER

for j in "${!max_tokens_per_batch_values[@]}"; do
for i in "${!trace_files[@]}"; do
    TRACE_FILE="${TRACES_FOLDER}/${trace_files[$i]}.json"
    test -f $TRACE_FILE || { echo "File $TRACE_FILE not found"; exit 1; }
    MAX_TOKENS_PER_BATCH=${max_tokens_per_batch_values[$j]}
    echo "Running $TRACE_FILE with $MAX_TOKENS_PER_BATCH tokens per batch"
    LOG_FILE="${OUTPUT_FOLDER}/incr_dec_${trace_files[$i]}_${MAX_TOKENS_PER_BATCH}_tokens_per_batch.log"
    rm $LOG_FILE || true
    ./inference/incr_decoding/incr_decoding \
        -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE \
        -llm-model $MODEL_NAME --fusion \
        -tensor-parallelism-degree $NGPUS \
        -prompt $TRACE_FILE \
        -output-folder $OUTPUT_FOLDER \
        --max-requests-per-batch $BATCH_SIZE \
        --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
        --max-sequence-length $MAX_SEQ_LEN \
        2>&1 | tee $LOG_FILE
done
done

