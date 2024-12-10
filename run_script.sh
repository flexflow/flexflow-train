#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/build"

export BUILD_TYPE=Release
../config/config.linux
make -j install

model_name=meta-llama/Llama-3.1-70B-Instruct
NGPUS=8
NCPUS=16
FSIZE=38000
ZSIZE=200000
# CSIZE=100000

# comment these lines in for debugging
# model_name=meta-llama/Llama-3.1-8B-Instruct
# NGPUS=8
# FSIZE=36000
# ZSIZE=30000
# CSIZE=100000


PROMPT_FILE=../inference/prompt/peft.json
MAX_SEQ_LEN=2048
tokens_per_batch=512

batch_size=8
output_fp=../inference/output/test.out


./inference/incr_decoding/incr_decoding \
    -ll:gpu $NGPUS -ll:cpu $NCPUS -ll:util $NCPUS \
    --log-instance-creation \
    -tensor-parallelism-degree $NGPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE\
    --fusion \
    --max-sequence-length $MAX_SEQ_LEN \
    --max-requests-per-batch $batch_size \
    --max-tokens-per-batch $tokens_per_batch \
    -llm-model $model_name \
    -prompt $PROMPT_FILE \
    -output-file ../inference/output/test.json \
    2>&1 | tee ../inference/output/test.log
