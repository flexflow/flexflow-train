import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the CSV file
def plot_fwd_overhead(filepath, model_name, tp_degree, bz, num_tokens_per_batch, ft_bwd_tokens):
    # Load the CSV file
    df = pd.read_csv(filepath)
    
    # Calculate step_time as difference between consecutive timestamps
    # Convert from microseconds to milliseconds (divide by 1000)
    df['step_time'] = df['timestamp'].diff() / 1000
    
    # Filter rows based on the specified conditions
    filtered_df = df[
        (df['num_decoding_tokens'] == 8) &
        (df['num_prefilling_tokens'] == 0) &
        (df['num_finetuning_fwd_tokens'] == 0) &
        (df['num_finetuning_bwd_tokens'] == 0)
    ]
    # Calculate statistics for step_time
    avg_step_time = filtered_df['step_time'].mean()
    std_step_time = filtered_df['step_time'].std()
    print(f"Step time: {avg_step_time:.3f} ± {std_step_time:.3f} ms ({len(filtered_df)} entries)")

    if num_tokens_per_batch == 128:
        values_of_interest = [0, 8, 24, 56, 120]
    elif num_tokens_per_batch == 256:
        values_of_interest = [0, 8, 24, 56, 120, 248]
    elif num_tokens_per_batch == 512:
        values_of_interest = [0, 8, 24, 56, 120, 248, 504]

    # Second analysis: Variable finetuning tokens
    filtered_df_2 = df[
        (df['is_warmup_step'] == 0) &
        (df['num_decoding_tokens'] == 8) &
        (df['num_prefilling_tokens'] == 0) &
        (df['num_finetuning_bwd_tokens'] == 0) &
        (df['num_finetuning_fwd_tokens'].isin(values_of_interest))
    ]
    filtered_df_2 = filtered_df_2[['num_finetuning_fwd_tokens', 'step_time']]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=filtered_df_2, 
                   x='num_finetuning_fwd_tokens', 
                   y='step_time',
                   alpha=0.6)
    
    plt.title(f'Step Time vs Number of Finetuning Forward Tokens\nModel: {model_name} (TP={tp_degree})\nBatch Size: {bz} - Max Tokens per Batch: {num_tokens_per_batch}\nBWD finetuning tokens: {ft_bwd_tokens}')
    plt.xlabel('Number of Finetuning Forward Tokens')
    plt.ylabel('Step Time (milliseconds)')
    
    # Add trend line
    avg_std_df = filtered_df_2.groupby('num_finetuning_fwd_tokens').agg(
        avg_step_time=('step_time', 'mean'),
        std_step_time=('step_time', 'std')
    ).reset_index()

    plt.errorbar(avg_std_df['num_finetuning_fwd_tokens'], 
                 avg_std_df['avg_step_time'], 
                 yerr=avg_std_df['std_step_time'], 
                 fmt='-o', 
                 color='red', 
                 ecolor='gray', 
                 elinewidth=2, 
                 capsize=4)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'./plots/overhead_test/fwd_overhead_{num_tokens_per_batch}.pdf', bbox_inches='tight')

    # plt.show()
    
def plot_bwd_overhead(filepath, model_name, tp_degree, bz, num_tokens_per_batch, ft_bwd_tokens):
    # Load the CSV file
    df = pd.read_csv(filepath)
    
    # Calculate step_time as difference between consecutive timestamps
    # Convert from microseconds to milliseconds (divide by 1000)
    df['step_time'] = df['timestamp'].diff() / 1000
    
    # Filter rows based on the specified conditions
    filtered_df = df[
        (df['num_decoding_tokens'] == 8) &
        (df['num_prefilling_tokens'] == 0) &
        (df['num_finetuning_fwd_tokens'] == 0) &
        (df['num_finetuning_bwd_tokens'] == 0)
    ]
    
    # Calculate statistics for step_time
    avg_step_time = filtered_df['step_time'].mean()
    std_step_time = filtered_df['step_time'].std()
    print(f"Step time: {avg_step_time:.3f} ± {std_step_time:.3f} ms ({len(filtered_df)} entries)")

    values_of_interest=[0, 1,2,4,8,16,32,64]

    # Second analysis: Variable finetuning tokens
    filtered_df_2 = df[
        (df['is_warmup_step'] == 0) &
        (df['num_decoding_tokens'] == 8) &
        (df['num_prefilling_tokens'] == 0) &
        (df['num_finetuning_fwd_tokens'] == 0) &
        (df['num_finetuning_bwd_tokens'] == 1024) &
        (df['num_bwd_layers'].isin(values_of_interest))
    ]
    filtered_df_2 = filtered_df_2[['num_bwd_layers', 'step_time']]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=filtered_df_2, 
                   x='num_bwd_layers', 
                   y='step_time',
                   alpha=0.6)
    
    plt.title(f'Step Time vs Number of BWD Finetuning Layers\nModel: {model_name} (TP={tp_degree})\nBatch Size: {bz} - Max Tokens per Batch: {num_tokens_per_batch}\nBWD finetuning tokens: {ft_bwd_tokens}')
    plt.xlabel('Number of BWD Finetuning Layers')
    plt.ylabel('Step Time (milliseconds)')
    
    # Add trend line
    avg_std_df = filtered_df_2.groupby('num_bwd_layers').agg(
        avg_step_time=('step_time', 'mean'),
        std_step_time=('step_time', 'std')
    ).reset_index()
    # print(avg_std_df['avg_step_time'])
    plt.errorbar(avg_std_df['num_bwd_layers'], 
                 avg_std_df['avg_step_time'], 
                 yerr=avg_std_df['std_step_time'], 
                 fmt='-o', 
                 color='red', 
                 ecolor='gray', 
                 elinewidth=2, 
                 capsize=4)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'./plots/overhead_test/bwd_overhead_{num_tokens_per_batch}.pdf', bbox_inches='tight')

    # plt.show()

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
        fp=f"./data/overhead_test/step_profiling_overhead_test_{model_name_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_8_max_tokens_per_batch_{tokens_per_batch}_arrival_rate_0.000000_num_warmup_requests_10.csv"
        
        plot_fwd_overhead(fp, model_name, tp_degree, bz, tokens_per_batch, ft_bwd_tokens)
        plot_bwd_overhead(fp, model_name, tp_degree, bz, tokens_per_batch, ft_bwd_tokens)