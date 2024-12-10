import pandas as pd
from math import ceil
from random import uniform
import json, argparse


def generate_arrival_rates(target_arrival_rate=10, debug_verbose=False):
  def get_splitwise_trace(trace_type="conv"):
    # Import Microsoft LLM 1 hour trace
    df_trace = pd.read_csv("https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_"+trace_type+".csv", parse_dates=["TIMESTAMP"])
    req_times = (pd.to_datetime(df_trace["TIMESTAMP"]).astype(int)//1000) # Timestamps are in microseconds
    req_times = req_times - req_times.min()
    req_times = req_times.tolist()
    return req_times
  
  req_times = get_splitwise_trace()

  microsec = 1000000
  avg_arrival_rate = len(req_times) / (req_times[-1]/float(microsec)) # Request per second. Computed that way to enforce working with numbers of reasonable orders of magnitude
  if debug_verbose:
    print("Avg arrival rate of original trace (req/s): ", avg_arrival_rate)
  scale_factor = float(target_arrival_rate) / avg_arrival_rate
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
  return arrival_times


if __name__ == '__main__':
  # Set up the argument parser
  parser = argparse.ArgumentParser(description='Generate and save a trace.')
  parser.add_argument('--arrival-rate', type=float, default=10.0, help='The target arrival rate for the trace.')
  parser.add_argument('--output-file', type=str, default='sharegpt.json', help='The path to the output file to save the trace.')

  # Parse the command-line arguments
  args = parser.parse_args()

  # Call the function with the user-provided arrival rate
  arrival_times = generate_arrival_rates(target_arrival_rate=args.arrival_rate, debug_verbose=True)
  print(f"Generated arrival times for a max of {len(arrival_times)} requests")
  with open(args.output_file, 'w+') as f:
    json.dump(arrival_times, f, indent=2)
