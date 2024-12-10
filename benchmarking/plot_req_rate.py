from collections import defaultdict, namedtuple
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

IncrDecExpKey = namedtuple('IncrDecExpKey', ['model', 'dataset', 'max_tokens_per_batch', 'arrival_rate'])



def get_tpot(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or decoding_step_idx is < 0
    df = df[(df["is_warmup_request"] == 0) & (df["decoding_step_idx"] >= 0)]
    group = df.groupby("request_guid", as_index=False)
    min_time = group["timestamp"].min()["timestamp"]
    max_time = group["timestamp"].max()["timestamp"]
    num_generated_tokens = group.size()["size"]
    tpots = (max_time - min_time) / num_generated_tokens / 1000
    # return mean and p99 of tpots
    return tpots.mean(), tpots.median(), tpots.quantile(0.99)

def get_throughput(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or request_step_idx is < 0
    df = df[(df["is_warmup_request"] == 0) & (df["decoding_step_idx"] >= 0)]
    # compute the throughput as the number of rows in the filtered dataframe (df) divided by the total time taken
    microsec_to_sec = 1_000_000
    total_time_sec = (df["timestamp"].max() - df["timestamp"].min()) / microsec_to_sec
    total_output_tokens = df.shape[0]
    return total_output_tokens / total_time_sec

def get_ttft(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    ttft = group.apply(lambda x: x[x["decoding_step_idx"] == 0]["timestamp"].values[0] - x[x["decoding_step_idx"] == -1]["timestamp"].values[0])/1000
    # convert to milliseconds from microseconds
    return ttft.mean()[1], ttft.median()[1], ttft.quantile(0.99)[1]

def get_queueing_time(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    microsec_to_sec = 1_000_000
    # in each group, find the difference between the timestampt at request_step_idx=-1 and the timestamp at request_step_idx=-2.
    queueing_time = group.apply(lambda x: x[x["decoding_step_idx"] == -1]["timestamp"].values[0] - x[x["decoding_step_idx"] == -2]["timestamp"].values[0])/microsec_to_sec
    return queueing_time.mean()[1], queueing_time.median()[1], queueing_time.quantile(0.99)[1]


def plot_tpot_vs_throughput(results_dict):
    """
    Create side-by-side plots of TPOT vs throughput for each dataset.
    """
    # Get unique datasets
    datasets = set(key.dataset for key in results_dict.keys())
    
    # Create a figure with subplots side by side
    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]  # Make axes iterable if there's only one dataset
    
    # Process data for each dataset
    for dataset, ax in zip(sorted(datasets), axes):
        # Group data by model for this dataset
        model_data = defaultdict(lambda: {'throughput': [], 'tpot_mean': [], 'tpot_p99': [], 'batch_sizes': []})
        
        # Process data for this dataset
        for key, df in results_dict.items():
            if key.dataset == dataset:
                model = key.model
                throughput = get_throughput(df)
                tpot_mean, tpot_median, tpot_p99 = get_tpot(df)
                
                model_data[model]['throughput'].append(throughput)
                model_data[model]['tpot_mean'].append(tpot_mean)
                model_data[model]['tpot_p99'].append(tpot_p99)
                model_data[model]['batch_sizes'].append(key.max_tokens_per_batch)
        
        # Plot for each model
        for model, data in model_data.items():
            # Sort all arrays based on throughput
            sort_idx = np.argsort(data['throughput'])
            throughput = np.array(data['throughput'])[sort_idx]
            tpot_mean = np.array(data['tpot_mean'])[sort_idx]
            tpot_p99 = np.array(data['tpot_p99'])[sort_idx]
            batch_sizes = np.array(data['batch_sizes'])[sort_idx]
            
            # Plot mean TPOT
            line_mean = ax.plot(throughput, tpot_mean, '-o', label=f'Mean TPOT', markersize=6)
            color = line_mean[0].get_color()
            
            # Plot P99 TPOT with same color but dashed line
            ax.plot(throughput, tpot_p99, '--o', label=f'P99 TPOT', 
                   color=color, markersize=6)
            
            # Add batch size annotations for both mean and p99 points
            for x, y_mean, y_p99, bs in zip(throughput, tpot_mean, tpot_p99, batch_sizes):
                # Annotate mean points
                ax.annotate(f'{bs}', (x, y_mean), xytext=(5, 5), 
                           textcoords='offset points', ha='left', fontsize=8,
                           color=color)
                # Annotate p99 points
                ax.annotate(f'{bs}', (x, y_p99), xytext=(5, 5), 
                           textcoords='offset points', ha='left', fontsize=8,
                           color=color)
        
        # Customize plot
        ax.set_xlabel('Output Throughput (tokens/sec)')
        ax.set_ylabel('TPOT (ms/token)')
        ax.set_title(f'Throughput-Latency characterization\n{model} (TP=4)\nBatch Size: 8\nDataset: {dataset}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'./plots/req_rate_test/throughput_vs_latency.pdf', bbox_inches='tight')
    plt.show()

def plot_ttft_bars(results_dict):
    # Get unique datasets
    datasets = set(key.dataset for key in results_dict.keys())
    
    # Create figure with subplots side by side
    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 6))
    if len(datasets) == 1:
        axes = [axes]
    
    # Process data for each dataset
    for dataset, ax in zip(sorted(datasets), axes):
        # Collect data for this dataset
        batch_sizes = set()
        model_data = defaultdict(lambda: {'batch_sizes': [], 'ttft_mean': [], 'ttft_p99': []})
        
        # Process data for this dataset
        for key, df in results_dict.items():
            if key.dataset == dataset:
                model = key.model
                ttft_mean, ttft_median, ttft_p99 = get_ttft(df)
                
                model_data[model]['batch_sizes'].append(key.max_tokens_per_batch)
                model_data[model]['ttft_mean'].append(ttft_mean)
                model_data[model]['ttft_p99'].append(ttft_p99)
                batch_sizes.add(key.max_tokens_per_batch)
        
        # Set up the plot
        x = np.arange(len(batch_sizes))  # Positions for bars
        width = 0.15  # Width of bars
        batch_sizes = sorted(batch_sizes)
        
        # Plot bars for each model
        for i, (model, data) in enumerate(model_data.items()):
            # Sort data by batch size
            sort_idx = np.argsort(data['batch_sizes'])
            ttft_mean = np.array(data['ttft_mean'])[sort_idx]
            ttft_p99 = np.array(data['ttft_p99'])[sort_idx]
            
            # Plot mean TTFT bars
            mean_pos = x - width * (len(model_data)/2 - i)
            bars_mean = ax.bar(mean_pos, ttft_mean, width, 
                             label=f'{model} (mean)',
                             alpha=0.8)
            
            # Plot p99 TTFT bars
            p99_pos = x + len(batch_sizes) + 0.5 - width * (len(model_data)/2 - i)
            bars_p99 = ax.bar(p99_pos, ttft_p99, width,
                             label=f'{model} (p99)',
                             alpha=0.8,
                             hatch='//')
            
            # Add value labels on top of bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.2f}',
                           ha='center', va='bottom', rotation=0,
                           fontsize=8)
            
            add_value_labels(bars_mean)
            add_value_labels(bars_p99)
        
        # Customize plot
        ax.set_ylabel('TTFT (ms)')
        # ax.set_title(f'{dataset} Dataset')
        ax.set_title(f'TTFT\n{model} (TP=4)\nBatch Size: 8\nDataset: {dataset}')
        
        # Set x-tick labels
        all_x = np.concatenate([x, x + len(batch_sizes) + 0.5])
        ax.set_xticks(all_x)
        all_labels = [str(bs) for bs in batch_sizes] * 2
        ax.set_xticklabels(all_labels)
        
        # Add group labels
        ax.text(np.mean(x), -0.1, 'Mean TTFT',
                ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.text(np.mean(x) + len(batch_sizes) + 0.5, -0.1, 'P99 TTFT',
                ha='center', va='top', transform=ax.get_xaxis_transform())
        
        # Add grid and legend
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add batch size label
        ax.set_xlabel('Tokens per batch')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'./plots/req_rate_test/ttft.pdf', bbox_inches='tight')
    plt.show()


def plot_queue_time_bars(results_dict):
    # Get unique datasets
    datasets = set(key.dataset for key in results_dict.keys())
    
    # Create figure with subplots side by side
    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 6))
    if len(datasets) == 1:
        axes = [axes]
    
    # Process data for each dataset
    for dataset, ax in zip(sorted(datasets), axes):
        # Collect data for this dataset
        batch_sizes = set()
        model_data = defaultdict(lambda: {'batch_sizes': [], 'queue_time_mean': [], 'queue_time_p99': []})
        
        # Process data for this dataset
        for key, df in results_dict.items():
            if key.dataset == dataset:
                model = key.model
                queue_time_mean, queue_time_median, queue_time_p99 = get_queueing_time(df)
                
                model_data[model]['batch_sizes'].append(key.max_tokens_per_batch)
                model_data[model]['queue_time_mean'].append(queue_time_mean)
                model_data[model]['queue_time_p99'].append(queue_time_p99)
                batch_sizes.add(key.max_tokens_per_batch)
        
        # Set up the plot
        x = np.arange(len(batch_sizes))  # Positions for bars
        width = 0.15  # Width of bars
        batch_sizes = sorted(batch_sizes)
        
        # Plot bars for each model
        for i, (model, data) in enumerate(model_data.items()):
            # Sort data by batch size
            sort_idx = np.argsort(data['batch_sizes'])
            queue_time_mean = np.array(data['queue_time_mean'])[sort_idx]
            queue_time_p99 = np.array(data['queue_time_p99'])[sort_idx]
            
            # Plot mean Queue Time bars
            mean_pos = x - width * (len(model_data)/2 - i)
            bars_mean = ax.bar(mean_pos, queue_time_mean, width, 
                             label=f'{model} (mean)',
                             alpha=0.8)
            
            # Plot p99 TTFT bars
            p99_pos = x + len(batch_sizes) + 0.5 - width * (len(model_data)/2 - i)
            bars_p99 = ax.bar(p99_pos, queue_time_p99, width,
                             label=f'{model} (p99)',
                             alpha=0.8,
                             hatch='//')
            
            # Add value labels on top of bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.2f}',
                           ha='center', va='bottom', rotation=0,
                           fontsize=8)
            
            add_value_labels(bars_mean)
            add_value_labels(bars_p99)
        
        # Customize plot
        ax.set_ylabel('Queue Time (s)')
        # ax.set_title(f'{dataset} Dataset')
        ax.set_title(f'Queueing Time\n{model} (TP=4)\nBatch Size: 8\nDataset: {dataset}')
        
        # Set x-tick labels
        all_x = np.concatenate([x, x + len(batch_sizes) + 0.5])
        ax.set_xticks(all_x)
        all_labels = [str(bs) for bs in batch_sizes] * 2
        ax.set_xticklabels(all_labels)
        
        # Add group labels
        ax.text(np.mean(x), -0.1, 'Mean Queue Time',
                ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.text(np.mean(x) + len(batch_sizes) + 0.5, -0.1, 'P99 Queue Time',
                ha='center', va='top', transform=ax.get_xaxis_transform())
        
        # Add grid and legend
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add batch size label
        ax.set_xlabel('Tokens per batch')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'./plots/req_rate_test/queueing_time.pdf', bbox_inches='tight')
    plt.show()

def get_dfs(models, datasets, max_tokens_per_batch, arrival_rates):
    incr_dec_req_profiling_fps = {
        IncrDecExpKey(model, dataset, max_tokens, arrival_rate): 
            f"./data/req_rate_test/inference_request_profiling_{dataset}_{model}_tensor_parallelism_4_max_requests_per_batch_8_max_tokens_per_batch_{max_tokens}_arrival_rate_{arrival_rate}_num_warmup_requests_10.csv"
        for model in models
        for dataset in datasets
        for max_tokens in max_tokens_per_batch
        for arrival_rate in arrival_rates
    }
    incr_dec_step_profiling_fps = {
        IncrDecExpKey(model, dataset, max_tokens, arrival_rate): 
            f"./data/req_rate_test/step_profiling_{dataset}_{model}_tensor_parallelism_4_max_requests_per_batch_8_max_tokens_per_batch_{max_tokens}_arrival_rate_{arrival_rate}_num_warmup_requests_10.csv"
        for model in models
        for dataset in datasets
        for max_tokens in max_tokens_per_batch
        for arrival_rate in arrival_rates
    }

    incr_dec_req_profiling_dfs = {
        key: pd.read_csv(val)
        for key, val in incr_dec_req_profiling_fps.items()
    }

    incr_dec_step_profiling_dfs = {
        key: pd.read_csv(val)
        for key, val in incr_dec_step_profiling_fps.items()
    }
    return incr_dec_req_profiling_dfs, incr_dec_step_profiling_dfs

if __name__ == "__main__":
    models=["meta-llama/llama-3.1-70b".replace("/", "_")]
    # datasets=["sharegpt", "wildchat"]
    # max_tokens_per_batch=[128, 256, 512]
    # arrival_rates = ["0.000000"]
    datasets=["wildchat"]
    max_tokens_per_batch=[128]
    arrival_rates = ["0.200000"]
    # inference_request_profiling_wildchat_meta-llama_llama-3.1-70b_tensor_parallelism_4_max_requests_per_batch_8_max_tokens_per_batch_128_arrival_rate__num_warmup_requests_10.csv
    # Change working directory to folder containing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    incr_dec_req_profiling_dfs, incr_dec_step_profiling_dfs = get_dfs(models, datasets, max_tokens_per_batch, arrival_rates)
    
    # Print sample statistics
    target_df = incr_dec_req_profiling_dfs[IncrDecExpKey(model='meta-llama_llama-3.1-70b', dataset='wildchat', max_tokens_per_batch=128, arrival_rate='0.200000')]
    tpot_mean, tpot_median, tpot_p99 = get_tpot(target_df)
    print(f"TPOT mean: {tpot_mean:.3f} ms/token, TPOT median: {tpot_median:.3f} ms/token, TPOT p99: {tpot_p99:.3f} ms/token")
    throughput = get_throughput(target_df)
    print(f"Throughput: {throughput:.3f} tokens/sec")
    ttft_mean, ttft_median, ttft_p99 = get_ttft(target_df)
    print(f"TTFT mean: {ttft_mean:.3f} ms, TTFT median: {ttft_median:.3f} ms, TTFT p99: {ttft_p99:.3f} ms")
    queue_time_mean, queue_time_median, queue_time_p99 = get_queueing_time(target_df)
    print(f"Queue time mean: {queue_time_mean:.3f} sec, Queue time median: {queue_time_median:.3f} sec, Queue time p99: {queue_time_p99:.3f} sec")

    # Make output directory
    if not os.path.exists('./plots/req_rate_test'):
        os.makedirs('./plots/req_rate_test')

    # Make plots
    plot_tpot_vs_throughput(incr_dec_req_profiling_dfs)
    plot_ttft_bars(incr_dec_req_profiling_dfs)
    plot_queue_time_bars(incr_dec_req_profiling_dfs)
