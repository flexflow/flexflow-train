/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/inference.h"
#include "flexflow/request_manager.h"
#include "models/falcon.h"
#include "models/llama.h"
#include "models/mpt.h"
#include "models/opt.h"
#include "models/starcoder.h"
#include <wordexp.h>

#include <nlohmann/json.hpp>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("llama");

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
  std::string output_folder_path;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      std::string &llm_model_name,
                      bool &use_full_precision,
                      bool &verbose,
                      bool &do_sample,
                      float &temperature,
                      float &topp,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      llm_model_name = std::string(argv[++i]);
      for (char &c : llm_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    // cache folder
    if (!strcmp(argv[i], "-cache-folder")) {
      paths.cache_folder_path = std::string(argv[++i]);
      continue;
    }
    // prompts
    if (!strcmp(argv[i], "-prompt")) {
      paths.prompt_file_path = std::string(argv[++i]);
      continue;
    }
    // output file
    if (!strcmp(argv[i], "-output-folder")) {
      paths.output_folder_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--use-full-precision")) {
      use_full_precision = true;
      continue;
    }
    // verbose logging to stdout
    if (!strcmp(argv[i], "--verbose")) {
      verbose = true;
      continue;
    }
    if (!strcmp(argv[i], "--do-sample")) {
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--temperature")) {
      temperature = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--topp")) {
      topp = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-requests-per-batch")) {
      max_requests_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tokens-per-batch")) {
      max_tokens_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-sequence-length")) {
      max_sequence_length = std::stoi(argv[++i]);
      continue;
    }
  }
  if (paths.cache_folder_path.empty()) {
    char const *ff_cache_path = std::getenv("FF_CACHE_PATH");
    paths.cache_folder_path = ff_cache_path ? std::string(ff_cache_path)
                                            : std::string("~/.cache/flexflow");
  }
  // Expand ~ to the home directory if needed
  wordexp_t p;
  wordexp(paths.cache_folder_path.c_str(), &p, 0);
  paths.cache_folder_path = p.we_wordv[0];
  wordfree(&p);
}

std::vector<Request> make_warmup_requests(int num_requests) {
  std::vector<Request> warmup_requests;
  for (int i = 0; i < num_requests; i++) {
    Request inference_req;
    inference_req.benchmarking_tokens = 512;
    inference_req.max_new_tokens = 30;
    inference_req.warmup = true;
    warmup_requests.push_back(inference_req);
  }
  return warmup_requests;
}

std::vector<Request> load_prompt_old(std::string trace_file_path) {
  int total_num_requests = 0;

  std::ifstream file_handle(trace_file_path);
  assert(file_handle.good() && "Prompt file does not exist.");
  nlohmann::json prompt_json = nlohmann::json::parse(file_handle,
                                  /*parser_callback_t */ nullptr,
                                  /*allow_exceptions */ true,
                                  /*ignore_comments */ true);

  std::vector<Request> requests;
  for (auto &prompt : prompt_json) {
    std::string text = prompt.get<std::string>();
    printf("Prompt[%d]: %s\n", total_num_requests, text.c_str());
    Request inference_req;
    inference_req.prompt = text;
    inference_req.max_length = 128;
    requests.push_back(inference_req);
    total_num_requests++;
  }
  return requests;
}

std::vector<Request> load_trace(std::string trace_file_path) {
  std::vector<Request> requests;
  
  std::ifstream file_handle(trace_file_path);
  assert(file_handle.good() && "Prompt file does not exist.");
  nlohmann::ordered_json prompt_json =
      nlohmann::ordered_json::parse(file_handle,
                                    /*parser_callback_t */ nullptr,
                                    /*allow_exceptions */ true,
                                    /*ignore_comments */ true);
  file_handle.close();
  auto &metadata = prompt_json["metadata"];
  
  for (auto &entry : prompt_json["entries"]) {
    int prompt_length = entry["prompt_length"];
    int response_length = entry["response_length"];
    std::string text = entry["prompt"];

    Request inference_req;
    // inference_req.prompt = text;
    inference_req.benchmarking_tokens = prompt_length;
    // inference_req.add_special_tokens = false;
    inference_req.max_new_tokens = response_length;
    requests.push_back(inference_req);
  }
  return requests;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  if (ffconfig.cpu_offload == false && ffconfig.quantization_type != DT_NONE) {
    assert(false && "Doesn't support quantization in non-offload mode");
  }
  FilePaths file_paths;
  std::string llm_model_name;
  bool use_full_precision = false;
  bool verbose = false;
  bool do_sample = false;
  float temperature = 0.0f;
  float topp = 0.0f;
  int max_requests_per_batch = 8;
  int max_tokens_per_batch = 256;
  int max_sequence_length = 2048;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   llm_model_name,
                   use_full_precision,
                   verbose,
                   do_sample,
                   temperature,
                   topp,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length);

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  std::string config_filepath = join_path(
      {file_paths.cache_folder_path, "configs", llm_model_name, "config.json"});
  std::string tokenizer_filepath =
      join_path({file_paths.cache_folder_path, "tokenizers", llm_model_name});
  std::string weights_filepath =
      join_path({file_paths.cache_folder_path,
                 "weights",
                 llm_model_name,
                 use_full_precision ? "full-precision" : "half-precision"});
  std::ifstream config_file_handle(config_filepath);
  if (!config_file_handle.good()) {
    std::cout << "Model config file " << config_filepath << " not found."
              << std::endl;
    assert(false);
  }
  json model_config = json::parse(config_file_handle,
                                  /*parser_callback_t */ nullptr,
                                  /*allow_exceptions */ true,
                                  /*ignore_comments */ true);
  ModelType model_type = ModelType::UNKNOWN;
  auto architectures = model_config["architectures"];
  for (auto const &str : architectures) {
    if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM") {
      model_type = ModelType::LLAMA;
      break;
    } else if (str == "OPTForCausalLM") {
      model_type = ModelType::OPT;
      break;
    } else if (str == "RWForCausalLM" || str == "FalconForCausalLM") {
      model_type = ModelType::FALCON;
      break;
    } else if (str == "GPTBigCodeForCausalLM") {
      model_type = ModelType::STARCODER;
      break;
    } else if (str == "MPTForCausalLM") {
      model_type = ModelType::MPT;
      break;
    }
  }
  int bos_token_id = model_config.find("bos_token_id") == model_config.end()
                         ? -1
                         : (int)model_config.at("bos_token_id");
  // parse eos token id, which can be either a single integer or an array of
  // integers. Convert to std::vector<int>
  std::vector<int> eos_token_ids;
  if (model_config.find("eos_token_id") != model_config.end()) {
    if (model_config["eos_token_id"].is_array()) {
      for (auto &eos_token_id : model_config["eos_token_id"]) {
        eos_token_ids.push_back(eos_token_id);
      }
    } else {
      eos_token_ids.push_back(model_config["eos_token_id"]);
    }
  } else {
    eos_token_ids.push_back(-1);
  }

  assert(model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");

  GenerationConfig generationConfig(do_sample, temperature, topp);
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_verbose(verbose);
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->register_tokenizer(
      model_type, bos_token_id, eos_token_ids, tokenizer_filepath);
  std::string output_filepath = join_path(
      {file_paths.output_folder_path, "output.log"});
  rm->register_output_filepath(output_filepath);

  FFModel model(ffconfig, ffconfig.cpu_offload);
  if (model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(model,
                              config_filepath,
                              weights_filepath,
                              INC_DECODING_MODE,
                              generationConfig,
                              use_full_precision);
  } else if (model_type == ModelType::OPT) {
    OPT::create_opt_model(model,
                          config_filepath,
                          weights_filepath,
                          INC_DECODING_MODE,
                          use_full_precision);
  } else if (model_type == ModelType::FALCON) {
    FALCON::create_falcon_model(model,
                                config_filepath,
                                weights_filepath,
                                INC_DECODING_MODE,
                                use_full_precision);
  } else if (model_type == ModelType::STARCODER) {
    STARCODER::create_starcoder_model(model,
                                      config_filepath,
                                      weights_filepath,
                                      INC_DECODING_MODE,
                                      generationConfig,
                                      use_full_precision);
  } else if (model_type == ModelType::MPT) {
    MPT::create_mpt_model(model,
                          config_filepath,
                          weights_filepath,
                          INC_DECODING_MODE,
                          generationConfig,
                          use_full_precision);
  } else {
    assert(false && "unknow model type");
  }

  assert(!model.config.enable_peft);

  rm->start_background_server(&model);
  

  std::vector<Request> warmup_requests = make_warmup_requests(10);
  std::vector<GenerationResult> warmup_result = model.generate(warmup_requests);
  std::cout << "----------warmup finished--------------" << std::endl;
  std::vector<Request> requests = load_trace(file_paths.prompt_file_path);
  std::vector<GenerationResult> result = model.generate(requests);
  std::cout << "----------inference finished--------------" << std::endl;

  // terminate the request manager by stopping the background thread
  rm->terminate_background_server();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  std::string dataset_name;
  // set dataset name to "wildchat" if the prompt file path contains "wildchat"
  if (file_paths.prompt_file_path.find("wildchat") != std::string::npos) {
    dataset_name = "wildchat";
  } else if (file_paths.prompt_file_path.find("sharegpt") != std::string::npos) {
    dataset_name = "sharegpt";
  } else {
    dataset_name = "unknown";
  }

  std::cout << "Saving profiling info..." << std::endl;
  rm->save_profiling_info_to_csv(file_paths.output_folder_path,
                                 dataset_name,
                                 llm_model_name,
                                 ffconfig.tensor_parallelism_degree,
                                 max_requests_per_batch,
                                 max_tokens_per_batch,
                                 0.0, // arrival rate
                                 10); // num_warmup_requests

  // free tokenizer space in memory
}

void FlexFlow::register_custom_tasks() {}
