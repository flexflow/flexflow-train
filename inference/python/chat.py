# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flexflow.serve as ff
import argparse, json, os
from types import SimpleNamespace


def get_configs():
    # Define sample configs
    ff_init_configs = {
        # required parameters
        "num_gpus": 8,
        "memory_per_gpu": 34000,
        "zero_copy_memory_per_node": 200000,
        # optional parameters
        "num_cpus": 16,
        "legion_utility_processors": 16,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": 8,
        "pipeline_parallelism_degree": 1,
        "offload": False,
        "offload_reserve_space_size": 8 * 1024,  # 8GB
        "use_4bit_quantization": False,
        "use_8bit_quantization": False,
        "enable_peft": False,
        "profiling": False,
        "benchmarking": False,
        "inference_debugging": False,
        "fusion": True,
    }
    llm_configs = {
        # required parameters
        "llm_model": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        # optional parameters
        "cache_path": os.environ.get("FF_CACHE_PATH", ""),
        "refresh_cache": False,
        "full_precision": False,
    }
    # Merge dictionaries
    ff_init_configs.update(llm_configs)
    return ff_init_configs


def main():
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)

    # Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
    ff.init(configs_dict)

    # Create the FlexFlow LLM
    ff_data_type = (
        ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    )
    llm = ff.LLM(
        configs.llm_model,
        data_type=ff_data_type,
        cache_path=configs.cache_path,
        refresh_cache=configs.refresh_cache,
    )

    # Compile the LLM for inference and load the weights into memory
    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        max_requests_per_batch=1,
        max_seq_length=2048,
        max_tokens_per_batch=256,
    )

    llm.start_server()

    nemotron_system = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are positive in nature."
    llama_generic_system = "You are a helpful an honest programming assistant."


    messages=[
        {"role": "system", "content": nemotron_system},
        {"role": "user", "content": "Is Rust better than Python?"},
    ]
    llm.generate(messages, max_new_tokens=1024)
    
    llm.stop_server()


if __name__ == "__main__":
    print("flexflow inference example (incremental decoding)")
    main()
