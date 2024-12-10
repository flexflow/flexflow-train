#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"


# MODEL_NAME="meta-llama/Llama-2-70b-hf"
# MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
# PEFT_MODEL_NAME="goliaro/llama-2-70b-hf-lora"
# NGPUS=8
# NCPUS=16
# FSIZE=38000
# ZSIZE=200000

# MODEL_NAME="meta-llama/Meta-Llama-3-8B"
# PEFT_MODEL_NAME="goliaro/llama-3-8b-lora-dolly"
# NGPUS=8
# NCPUS=16
# FSIZE=30000
# ZSIZE=30000

MODEL_NAME="JackFram/llama-160m"
PEFT_MODEL_NAME="goliaro/llama-160m-lora"
NGPUS=4
NCPUS=16
FSIZE=30000
ZSIZE=20000

PROMPT_FILE="/usr/FlexFlow/inference/prompt/sharegpt.json"
FINETUNING_FILE="/usr/FlexFlow/inference/prompt/finetuning_benchmarking.json"
OUTPUT_FOLDER="/usr/FlexFlow/inference/output"
# OUTPUT_FILE="/usr/FlexFlow/inference/output/test.json"
LOG_FILE="/usr/FlexFlow/inference/output/test.log"
MAX_SEQ_LEN=2900
MAX_TOKENS_PER_BATCH=512
BATCH_SIZE=8
MAX_TRAINING_STEPS=1000

reset
make -j install

python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id=\"${MODEL_NAME}\", allow_patterns=\"*.safetensors\", max_workers=30)"
python ../inference/utils/download_hf_model.py $MODEL_NAME
python ../inference/utils/download_peft_model.py $PEFT_MODEL_NAME
 
# mkdir -p ../inference/prompt
# cp ../benchmarking/sharegpt.json ../inference/prompt/sharegpt.json
# echo '["Two things are infinite: "]' > ../inference/prompt/peft.json
# echo '["“Two things are infinite: the universe and human stupidity; and I'\''m not sure about the universe.”"]' > ../inference/prompt/peft_dataset.json
# Create output folder
# mkdir -p ../inference/output

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/usr/FlexFlow/inference/output/nccl2.log

export LEGION_BACKTRACE=1
# export FF_DEBG_NO_WEIGHTS=1
# export CUDA_VISIBLE_DEVICES=1,2,3,4

# ./inference/incr_decoding/incr_decoding \
#     -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
#     -ll:fsize $FSIZE -ll:zsize $ZSIZE \
#     --log-instance-creation \
#     -llm-model $MODEL_NAME --fusion \
#     -prompt $PROMPT_FILE \
#     -tensor-parallelism-degree $NGPUS \
#     -output-file ../inference/output/test.json \
#     --max-requests-per-batch $BATCH_SIZE --max-tokens-per-batch $MAX_TOKENS_PER_BATCH --max-sequence-length $MAX_SEQ_LEN \
#     2>&1 | tee ../inference/output/test.log
rm $LOG_FILE $OUTPUT_FILE || true
./inference/peft/peft \
    -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE \
    -llm-model $MODEL_NAME --fusion  \
    -enable-peft -peft-model $PEFT_MODEL_NAME \
    -finetuning-dataset $FINETUNING_FILE \
    -prompt $PROMPT_FILE \
    -tensor-parallelism-degree $NGPUS \
    -output-folder $OUTPUT_FOLDER \
    --max-requests-per-batch $BATCH_SIZE \
    --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
    --max-sequence-length $MAX_SEQ_LEN \
    --max-training-steps $MAX_TRAINING_STEPS \
    2>&1 | tee $LOG_FILE

# -lg:prof 1 -lg:prof_logfile prof_%.gz --verbose --inference-debugging --log-instance-creation \\