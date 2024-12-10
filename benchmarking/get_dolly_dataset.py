import json, os
from datasets import load_dataset
from transformers import AutoTokenizer

def create_dataset(model_name, num_entries, output_filepath, seed):
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    finetuning_data = []
    lengths=[]
    for row in dataset:
        if len(finetuning_data) == num_entries:
            break
        if ("open_qa" in row['category'] or "closed_qa" in row['category']) and len(row['context']) == 0:
            prompt = row['instruction']
            response = row['response']
            templated_entry = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                add_generation_prompt=False,
                tokenize=False)
            finetuning_data.append(templated_entry)
            lengths.append(len(tokenizer(templated_entry)["input_ids"]))
    with open(output_filepath, 'w') as file:
        json.dump(finetuning_data, file, indent=2)
    print("Max length:", max(lengths))
    print("Min length:", min(lengths))
    print("Avg length:", sum(lengths)/len(lengths))

if __name__ == "__main__":
    # change working directory to the root of the project
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "inference", "prompt"))

    num_entries=1000
    seed = 42
    model_name="meta-llama/Llama-3.1-70B-Instruct"
    output_filepath="dolly.json"

    create_dataset(model_name, num_entries, output_filepath, seed)
