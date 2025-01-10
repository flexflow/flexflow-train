import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

expected_fwd_tokens = {
    128: [0, 8, 24, 56, 120],
    256: [0, 8, 24, 56, 120, 248],
    512: [0, 8, 24, 56, 120, 248, 504]
}

expected_bwd_layers = {
    128: [0, 1, 2, 4, 8, 16, 32, 64],
    256: [0, 1, 2, 4, 8, 16, 32, 64],
    512: [0, 1, 2, 4, 8, 16, 32, 64]
}

def get_run_idx(max_tokens_per_batch, num_fwd_finetuning_tokens, num_bwd_layers):
    if max_tokens_per_batch not in [128, 256, 512]:
        raise ValueError("Invalid value for max_tokens_per_batch")
    
    fwd_idx = expected_fwd_tokens[max_tokens_per_batch].index(num_fwd_finetuning_tokens)
    bwd_idx = expected_bwd_layers[max_tokens_per_batch].index(num_bwd_layers)
    return 1 + fwd_idx * len(expected_bwd_layers[max_tokens_per_batch]) + bwd_idx

def get_req_guids(max_tokens_per_batch, num_fwd_finetuning_tokens, num_bwd_layers):
    num_warmup_requests=11
    max_requets_per_batch=8

    fwd_idx = expected_fwd_tokens[max_tokens_per_batch].index(num_fwd_finetuning_tokens)
    bwd_idx = expected_bwd_layers[max_tokens_per_batch].index(num_bwd_layers)
    
    # Add warmup requests
    start_guid = num_warmup_requests
    # Add requests from runs with 0 fwd tokens, where each group has 8 requests
    if fwd_idx == 0:
        start_guid += max_requets_per_batch * bwd_idx
    else:
        start_guid += max_requets_per_batch * len(expected_bwd_layers[max_tokens_per_batch])
        # Add requests from runs with >= 1 fwd tokens
        start_guid += ((fwd_idx-1) * len(expected_bwd_layers[max_tokens_per_batch]) + bwd_idx) * (max_requets_per_batch+1)
    guids=[1000000+x for x in range(start_guid, start_guid+max_requets_per_batch)]
    return guids

def get_tpots(df_):
    df = df_.copy()
    df = df[(df["is_warmup_request"] == 0) & (df["decoding_step_idx"] >= 0)]
    group = df.groupby("request_guid", as_index=False)
    min_time = group["timestamp"].min()["timestamp"]
    max_time = group["timestamp"].max()["timestamp"]
    num_generated_tokens = group.size()["size"]
    tpots = (max_time - min_time) / num_generated_tokens / 1000
    return tpots.mean(), tpots.median(), tpots.quantile(0.99)

def plot_fwd_overhead(filepath, model_name, tp_degree, bz, num_tokens_per_batch, ft_bwd_tokens):
    # Load the CSV file
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    # Plot one line for each bwd token value
    for fwd_tokens in expected_fwd_tokens[num_tokens_per_batch]:
        for bwd_layers in expected_bwd_layers[num_tokens_per_batch]:
            run_idx = get_run_idx(num_tokens_per_batch, fwd_tokens, bwd_layers)
            print(f"Run {run_idx}: {fwd_tokens} fwd tokens, {bwd_layers} bwd layers")

    for bwd_layers in expected_bwd_layers[num_tokens_per_batch]:
        latencies=[]
        for fwd_tokens in expected_fwd_tokens[num_tokens_per_batch]:
            run_idx = get_run_idx(num_tokens_per_batch, fwd_tokens, bwd_layers)
            grouped = df[df['run_idx'] == run_idx].copy()
            grouped['step_time'] = grouped['timestamp'].diff() / 1000
            filtered_df = grouped[
                (grouped['num_decoding_tokens'] == 8) &
                (grouped['num_prefilling_tokens'] == 0) &
                (grouped['num_finetuning_fwd_tokens'] == fwd_tokens) &
                (grouped['num_finetuning_bwd_tokens'] == 0)
            ]
            latencies.append(filtered_df['step_time'].mean())
        plt.plot(expected_fwd_tokens[num_tokens_per_batch], latencies, marker='o', label=f'Backward layers = {bwd_layers}')

    plt.xticks(expected_fwd_tokens[num_tokens_per_batch]) 
    plt.title(f"Batch Step Time vs Number of Forward Finetuning Tokens\nModel: {model_name} (TP={tp_degree})\nBatch Size: {bz} - Max Tokens per Batch: {num_tokens_per_batch}\nBWD finetuning tokens: {ft_bwd_tokens}")
    plt.xlabel('Number of forward finetuning tokens')
    plt.ylabel('Average step time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'./plots/overhead_test/fwd_overhead_{num_tokens_per_batch}.pdf', bbox_inches='tight')
    

def plot_bwd_overhead(filepath, model_name, tp_degree, bz, num_tokens_per_batch, ft_bwd_tokens):
    # Load the CSV file
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    # # Plot one line for each bwd token value
    # for fwd_tokens in expected_fwd_tokens[num_tokens_per_batch]:
    #     for bwd_layers in expected_bwd_layers[num_tokens_per_batch]:
    #         run_idx = get_run_idx(num_tokens_per_batch, fwd_tokens, bwd_layers)
    #         print(f"Run {run_idx}: {fwd_tokens} fwd tokens, {bwd_layers} bwd layers")

    for fwd_tokens in expected_fwd_tokens[num_tokens_per_batch]:
        latencies=[]
        for bwd_layers in expected_bwd_layers[num_tokens_per_batch]:
            run_idx = get_run_idx(num_tokens_per_batch, fwd_tokens, bwd_layers)
            grouped = df[df['run_idx'] == run_idx].copy()
            grouped['step_time'] = grouped['timestamp'].diff() / 1000
            filtered_df = grouped[
                (grouped['num_decoding_tokens'] == 8) &
                (grouped['num_prefilling_tokens'] == 0) &
                (grouped['num_finetuning_fwd_tokens'] == 0) &
                (grouped['num_finetuning_bwd_tokens'] == 1024) &
                (grouped['num_bwd_layers']== bwd_layers)
            ]
            latencies.append(filtered_df['step_time'].mean())
        plt.plot(expected_bwd_layers[num_tokens_per_batch], latencies, marker='o', label=f'Forward finetuning tokens = {fwd_tokens}')

    plt.xticks(expected_bwd_layers[num_tokens_per_batch]) 
    plt.title(f"Batch Step Time vs Number of Backward Finetuning Layers\nModel: {model_name} (TP={tp_degree})\nBatch Size: {bz} - Max Tokens per Batch: {num_tokens_per_batch}\nBWD finetuning tokens: {ft_bwd_tokens}")
    plt.xlabel('Number of backward finetuning layers')
    plt.ylabel('Average step time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'./plots/overhead_test/bwd_overhead_{num_tokens_per_batch}.pdf', bbox_inches='tight')
    
def plot_tpots(filepath, model_name, tp_degree, bz, num_tokens_per_batch, ft_bwd_tokens):
    # Load the CSV file
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    # Plot one line for each bwd token value
    # for fwd_tokens in expected_fwd_tokens[num_tokens_per_batch]:
    #     for bwd_layers in expected_bwd_layers[num_tokens_per_batch]:
    #         run_idx = get_run_idx(num_tokens_per_batch, fwd_tokens, bwd_layers)
    #         print(f"Run {run_idx}: {fwd_tokens} fwd tokens, {bwd_layers} bwd layers")

    for bwd_layers in expected_bwd_layers[num_tokens_per_batch]:
        latencies=[]
        for fwd_tokens in expected_fwd_tokens[num_tokens_per_batch]:
            guids = get_req_guids(num_tokens_per_batch, fwd_tokens, bwd_layers)
            filtered = df[df['request_guid'].isin(guids)]
            tpots = get_tpots(filtered)
            latencies.append(tpots[0])
        plt.plot(expected_fwd_tokens[num_tokens_per_batch], latencies, marker='o', label=f'Backward layers = {bwd_layers}')

    plt.xticks(expected_fwd_tokens[num_tokens_per_batch]) 
    plt.title(f"TPOT vs Number of Forward Finetuning Tokens\nModel: {model_name} (TP={tp_degree})\nBatch Size: {bz} - Max Tokens per Batch: {num_tokens_per_batch}\nBWD finetuning tokens: {ft_bwd_tokens}")
    plt.xlabel('Number of forward finetuning tokens')
    plt.ylabel('Average step time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'./plots/overhead_test/tpot_{num_tokens_per_batch}.pdf', bbox_inches='tight')


if __name__ == "__main__":

    # Change working directory to folder containing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Make plots directory if it doesn't exist
    if not os.path.exists('./plots/overhead_test'):
        os.makedirs('./plots/overhead_test')

    model_name="meta-llama/Llama-3.1-70B"
    model_name_=model_name.replace("/", "_").lower()
    tp_degree=4
    ft_bwd_tokens=1024
    bz=8

    for tokens_per_batch in [128, 256, 512]:
        fp_step=f"./data/overhead_test/step_profiling_overhead_test_{model_name_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_8_max_tokens_per_batch_{tokens_per_batch}_arrival_rate_0.000000_num_warmup_requests_10.csv"
        fp_req=f"./data/overhead_test/inference_request_profiling_overhead_test_{model_name_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_8_max_tokens_per_batch_{tokens_per_batch}_arrival_rate_0.000000_num_warmup_requests_10.csv"
        plot_fwd_overhead(fp_step, model_name, tp_degree, bz, tokens_per_batch, ft_bwd_tokens)
        plot_bwd_overhead(fp_step, model_name, tp_degree, bz, tokens_per_batch, ft_bwd_tokens)
        plot_tpots(fp_req, model_name, tp_degree, bz, tokens_per_batch, ft_bwd_tokens)