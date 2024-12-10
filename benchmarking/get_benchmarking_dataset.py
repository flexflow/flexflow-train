import json, os
from transformers import AutoTokenizer

def create_test_dataset(model_name, num_entries, num_tokens_per_entry, output_filepath):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = [5466 for _ in range(num_tokens_per_entry-1)]
    detokenized = tokenizer.decode(input_ids)
    print(detokenized)
    print(len(tokenizer(detokenized)["input_ids"]))
    assert(len(tokenizer(detokenized)["input_ids"]) == num_tokens_per_entry)
    finetuning_data = [detokenized for _ in range(num_entries)]
    with open(output_filepath, 'w') as file:
        json.dump(finetuning_data, file, indent=2)

if __name__ == "__main__":
    # change working directory to the root of the project
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "inference", "prompt"))

    num_entries=1000
    num_tokens_per_entry=1023
    model_name="meta-llama/Llama-3.1-70B-Instruct"
    output_filepath="finetuning_benchmarking.json"

    create_test_dataset(model_name, num_entries, num_tokens_per_entry, output_filepath)
