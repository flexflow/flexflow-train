from dataclasses import asdict, dataclass, field
import json, os, random, requests, argparse
from tqdm.asyncio import tqdm
from typing import List, Optional
from collections import OrderedDict
from transformers import AutoTokenizer
import pandas as pd
from math import ceil
from random import uniform
import numpy as np

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

@dataclass
class TraceEntry:
    prompt: str
    response: str
    prompt_length: int
    response_length: int
    arrival_time: int

@dataclass
class TraceMetadata:
    num_warmup_requests: int
    avg_entries_per_partition: float
    max_prompt_length: int
    min_prompt_length: int
    avg_prompt_length: float
    max_response_length: int
    min_response_length: int
    avg_response_length: float
    max_total_length: int
    trace_type: str
    arrival_rate: float

@dataclass
class Trace:
    entries: List[TraceEntry] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=lambda: TraceMetadata(0, 0, 0, 0, 0, 0, 0, 0, 0, "offline", 0.0))

def generate_arrival_rates_splitwise(n, target_arrival_rate_sec, seed):
    def get_splitwise_trace(trace_type="conv"):
        # Import Microsoft LLM 1 hour trace
        df_trace = pd.read_csv("https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_"+trace_type+".csv", parse_dates=["TIMESTAMP"])
        req_times = (pd.to_datetime(df_trace["TIMESTAMP"]).astype(int)//1000) # Timestamps are in microseconds
        req_times = req_times - req_times.min()
        req_times = req_times.tolist()
        return req_times
  
    debug_verbose = True
    req_times = get_splitwise_trace()

    np.random.seed(seed)
    random.seed(seed)

    microsec = 1000000
    avg_arrival_rate = len(req_times) / (req_times[-1]/float(microsec)) # Request per second. Computed that way to enforce working with numbers of reasonable orders of magnitude
    if debug_verbose:
        print("Avg arrival rate of original trace (req/s): ", avg_arrival_rate)
    scale_factor = float(target_arrival_rate_sec) / avg_arrival_rate
    if debug_verbose:
        print("Scale factor to obtain target arrival rate: ", scale_factor)

    # Buckets are 1 second timeframes
    nb_buckets = ceil(req_times[-1] / microsec)
    j = 0
    # print("Number of buckets: ", nb_buckets)
    bucket_sizes=[]
    for i in range(nb_buckets):
        bucket_size = 0
        while(j < len(req_times) and req_times[j] >= i*microsec and req_times[j] < (i+1)*microsec):
            bucket_size += 1
            j += 1
        bucket_size = bucket_size*scale_factor
        prob = bucket_size - int(bucket_size)
        bucket_size = int(bucket_size) + int(uniform(0, 1) <= prob)
        bucket_sizes.append(bucket_size)

    arrival_times = []
    for arrival_time, num_requests in enumerate(bucket_sizes):
        for i in range(num_requests):
            arrival_times.append(arrival_time)
    if len(arrival_times) > n:
        arrival_times = arrival_times[:n]
    elif len(arrival_times) < n:
        raise RuntimeError(f"Error: not enough buckets ({nb_buckets}) in scaled trace to generate arrival times for all requests ({n})")
    return arrival_times

def generate_poisson_arrivals(n, target_arrival_rate_sec, seed):
    """
    Generate arrival times for n requests following a Poisson process.
    
    Parameters:
    n (int): Number of requests to generate
    arrival_rate (float): Average arrival rate (requests per second)
    
    Returns:
    numpy.ndarray: Array of arrival times in seconds
    """
    np.random.seed(seed)
    random.seed(seed)

    # Generate n exponentially distributed inter-arrival times
    # For a Poisson process, inter-arrival times follow exponential distribution
    inter_arrival_times = np.random.exponential(scale=1/target_arrival_rate_sec, size=n)
    
    # Calculate cumulative sum to get arrival times
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Round to 6 decimal places for practical purposes (microsecond precision)
    arrival_times = np.round(arrival_times, decimals=6)
    
    return arrival_times

def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename

def get_warmup_entries(model_name: str, num_warmup_requests: int) -> List[TraceEntry]:
    """
    Get a list of warmup entries for a model.
    
    Args:
    model_name (str): The name of the model.
    num_warmup_requests (int): The number of warmup requests to generate.
    
    Returns:
    List[TraceEntry]: A list of warmup entries.
    """
    warmup_entries = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for i in range(num_warmup_requests):
        prompt = "Hello, how are you?"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        response = "I'm doing well, thank you for asking."
        prompt_length = len(tokenizer(prompt)["input_ids"])
        response_length = len(tokenizer(response)["input_ids"])
        warmup_entries.append(TraceEntry(prompt, response, prompt_length, response_length, 0))
    return warmup_entries

def build_trace(model_name: str, 
                num_entries: int, 
                max_length: int, 
                seed: int, 
                trace_type: str = "offline",
                arrival_rate: float = 0.0,
                apply_chat_template: bool = False):
    # Download sharegpt if necessary
    dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f, object_pairs_hook=OrderedDict)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
        if data["conversations"][0]["from"] == "human" and data["conversations"][1]["from"] == "gpt"
    ]

    # Shuffle the dataset.
    random.seed(seed)
    random.shuffle(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    trace = Trace()
    trace_metadata = TraceMetadata(
        num_warmup_requests=0,
        avg_entries_per_partition=0,
        max_prompt_length=0,
        min_prompt_length=float("inf"),
        avg_prompt_length=0,
        max_response_length=0,
        min_response_length=float("inf"),
        avg_response_length=0,
        max_total_length=0,
        trace_type=trace_type,
        arrival_rate=arrival_rate
    )

    arrival_times = num_entries*[0.0]
    if trace_type == "poisson":
        arrival_times = generate_poisson_arrivals(num_entries, arrival_rate, seed)
    elif trace_type == "splitwise":
        arrival_times = generate_arrival_rates_splitwise(num_entries, arrival_rate, seed)
    assert(len(arrival_times) == num_entries)

    
    for i in tqdm(range(len(dataset))):
        if len(trace.entries) == num_entries:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        response = dataset[i][1]
        prompt_length = len(tokenizer(prompt)["input_ids"])
        response_length = len(tokenizer(response)["input_ids"])
        if prompt_length + response_length > max_length:
            continue
        new_entry = TraceEntry(prompt, response, prompt_length, response_length, arrival_times[len(trace.entries)])
        trace.entries.append(new_entry)
        trace_metadata.max_prompt_length = max(trace_metadata.max_prompt_length, prompt_length)
        trace_metadata.min_prompt_length = min(trace_metadata.min_prompt_length, prompt_length)
        trace_metadata.avg_prompt_length += prompt_length
        trace_metadata.max_response_length = max(trace_metadata.max_response_length, response_length)
        trace_metadata.min_response_length = min(trace_metadata.min_response_length, response_length)
        trace_metadata.avg_response_length += response_length
        trace_metadata.max_total_length = max(trace_metadata.max_total_length, prompt_length + response_length)
    trace_metadata.avg_prompt_length /= len(trace.entries)
    trace_metadata.avg_response_length /= len(trace.entries)
    trace_metadata.avg_entries_per_partition = len(trace.entries)
    trace_metadata.arrival_rate = arrival_rate

    trace.metadata = trace_metadata

    return trace

def save_trace(trace: Trace, output_path: str):
    """
    Save a Trace instance to a JSON file.
    
    Args:
    trace (Trace): The trace to save.
    output_path (str): The path where the JSON file will be saved.
    """
    # Convert the Trace instance to a dictionary
    trace_dict = asdict(trace)
    
    # Save the dictionary as a JSON file
    with open(output_path, 'w') as f:
        json.dump(trace_dict, f, indent=2)
    
    print(f"Trace saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WildChat trace")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Model name")
    parser.add_argument("-m", "--max-length", type=int, default=5000, help="Maximum prompt + response length")
    parser.add_argument("-n", "--num_entries", type=int, default=250, help="Number of entries")
    parser.add_argument("-s", "--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("-o", "--output_file", type=str, default="./traces/sharegpt.json", help="Output file name")
    parser.add_argument("-t", "--trace-type", type=str, choices=["offline", "poisson", "splitwise"], default="offline", help="Arrival Times Trace Type")
    parser.add_argument("-a", "--arrival-rate", type=float, default=0.0, help="Arrival Rate")
    args = parser.parse_args()

    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    trace = build_trace(args.model_name, 
                        args.num_entries, 
                        args.max_length, 
                        args.seed,
                        trace_type=args.trace_type,
                        arrival_rate=args.arrival_rate,
                        apply_chat_template=False)
    print(trace.metadata)
    # Save prompts list to a json file
    num_above_2048 = 0
    for entry in trace.entries:
        if entry.prompt_length + entry.response_length > 2048:
            num_above_2048 += 1
    print(f"Number of entries above 2048 tokens: {num_above_2048}")
    save_trace(trace, args.output_file)
