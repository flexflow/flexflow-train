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

#include <iostream>
#include <vector>
#include <cmath>

#include <nlohmann/json.hpp>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("llama");

struct FilePaths {
  std::string cache_folder_path;
  std::string output_folder_path;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      std::string &llm_model_name,
                      std::string &peft_model_name,
                      bool &use_full_precision,
                      bool &verbose,
                      bool &enable_peft,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length,
                      std::vector<int> &num_layers_per_finetuning_step,
                      std::vector<int> &max_fwd_finetuning_tokens) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      llm_model_name = std::string(argv[++i]);
      for (char &c : llm_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    if (!strcmp(argv[i], "-enable-peft")) {
      enable_peft = true;
      continue;
    }
    if (!strcmp(argv[i], "-peft-model")) {
      peft_model_name = std::string(argv[++i]);
      for (char &c : peft_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    // cache folder
    if (!strcmp(argv[i], "-cache-folder")) {
      paths.cache_folder_path = std::string(argv[++i]);
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
    if (!strcmp(argv[i], "--num-layers-per-finetuning-step")) {
      std::string layers_str = std::string(argv[++i]);
      std::stringstream ss(layers_str);
      std::string item;
      while (std::getline(ss, item, ',')) {
        num_layers_per_finetuning_step.push_back(std::stoi(item));
      }
      // std::cout << "ARG num_layers_per_finetuning_step: ";
      // for (int num_layers : num_layers_per_finetuning_step) {
      //   std::cout << num_layers << " ";
      // }
      // std::cout << std::endl;
      continue;
    }
    if (!strcmp(argv[i], "--max-fwd-finetuning-tokens")) {
      std::string tokens_str = std::string(argv[++i]);
      std::stringstream ss(tokens_str);
      std::string item;
      while (std::getline(ss, item, ',')) {
        max_fwd_finetuning_tokens.push_back(std::stoi(item));
      }
      // std::cout << "ARG max_fwd_finetuning_tokens: ";
      // for (int num_tokens : max_fwd_finetuning_tokens) {
      //   std::cout << num_tokens << " ";
      // }
      // std::cout << std::endl;
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

std::vector<Request> make_warmup_requests(int num_inf_request, int num_finetuning_steps, PEFTModelID *peft_model_id) {
  std::vector<Request> warmup_requests;

  for (int i = 0; i < num_inf_request; i++) {
    Request inference_req;
    inference_req.benchmarking_tokens = 512;
    inference_req.max_new_tokens = 30;
    inference_req.warmup = true;
    warmup_requests.push_back(inference_req);
  }
  Request finetuning_req;
  finetuning_req.req_type = RequestType::REQ_FINETUNING;
  finetuning_req.benchmarking_tokens = 1024;
  finetuning_req.add_special_tokens = false;
  finetuning_req.max_length = 1024;
  finetuning_req.warmup = true;
  finetuning_req.peft_model_id = (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
  finetuning_req.peft_finetuning_info.max_training_steps = num_finetuning_steps;
  warmup_requests.push_back(finetuning_req);
  return warmup_requests;
}

std::vector<Request> make_requests(int max_requests_per_batch, 
                                  int finetuning_entry_size, 
                                  int max_fwd_tokens_per_batch, 
                                  int tot_llm_layers,
                                  int bwd_layers_per_step,
                                  PEFTModelID *peft_model_id) {
  if (bwd_layers_per_step == 0) {
    bwd_layers_per_step = tot_llm_layers;
  }
  std:vector<Request> requests;
  int target_num_steps = 10;
  if (max_fwd_tokens_per_batch > 0) {
    target_num_steps += (finetuning_entry_size + max_fwd_tokens_per_batch - 1) / max_fwd_tokens_per_batch + 
                                  (tot_llm_layers + bwd_layers_per_step - 1) / bwd_layers_per_step;
  }
  for (int i = 0; i < max_requests_per_batch; i++) {
    Request inference_req;
    inference_req.benchmarking_tokens = 1;
    inference_req.add_special_tokens = false;
    inference_req.max_new_tokens = target_num_steps;
    inference_req.warmup = false;
    requests.push_back(inference_req);
  }
  if (max_fwd_tokens_per_batch > 0) {
    Request finetuning_req;
    finetuning_req.req_type = RequestType::REQ_FINETUNING;
    finetuning_req.add_special_tokens = false;
    finetuning_req.benchmarking_tokens = finetuning_entry_size;
    finetuning_req.max_length = finetuning_entry_size;
    finetuning_req.peft_model_id = (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
    finetuning_req.peft_finetuning_info.max_training_steps = 1;
    finetuning_req.warmup = false;
    requests.push_back(finetuning_req);
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
  std::string llm_model_name, peft_model_name;
  bool use_full_precision = false;
  bool verbose = false;
  bool do_sample = false;
  bool enable_peft = false;
  float temperature = 0.0f;
  float topp = 0.0f;
  int max_requests_per_batch = 8;
  int max_tokens_per_batch = 256;
  int max_sequence_length = 2048;
  int max_training_steps = -1;
  bool enable_peft_finetuning = true;
  std::vector<int> num_layers_per_finetuning_step;
  std::vector<int> max_fwd_finetuning_tokens;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   llm_model_name,
                   peft_model_name,
                   use_full_precision,
                   verbose,
                   enable_peft,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   num_layers_per_finetuning_step,
                   max_fwd_finetuning_tokens);
  // std::cout << "max_fwd_finetuning_tokens: ";
  // for (int num_tokens : max_fwd_finetuning_tokens) {
  //   std::cout << num_tokens << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "num_layers_per_finetuning_step: ";
  // for (int num_layers : num_layers_per_finetuning_step) {
  //   std::cout << num_layers << " ";
  // }
  // std::cout << std::endl;
  
  
  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  ffconfig.enable_peft_finetuning = enable_peft_finetuning;

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
  if (!enable_peft) {
    std::cerr << "Running PEFT script with PEFT not enabled" << std::endl;
    assert(false);
  }
  if (enable_peft && peft_model_name.empty()) {
    std::cout << "PEFT enabled, but no PEFT model id passed" << std::endl;
    assert(false);
  } else if (!enable_peft && !peft_model_name.empty()) {
    std::cout << "PEFT model id passed, but PEFT is not enabled" << std::endl;
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

  // load PEFT config
  LoraLinearConfig peft_config =
      peft_model_name.empty()
          ? LoraLinearConfig::EmptyConfig
          : LoraLinearConfig(file_paths.cache_folder_path, peft_model_name);

  LoraOptimizerConfig *optim_config = nullptr;
  if (enable_peft_finetuning) {
    // float sgd_learning_rate = 2e-1;
    float sgd_learning_rate = 0.001f;
    optim_config = new LoraSGDOptimizerConfig(sgd_learning_rate);
  }
  LoraLinearConfig peft_config_finetuning =
      !enable_peft_finetuning
          ? LoraLinearConfig::EmptyConfig
          : LoraLinearConfig(file_paths.cache_folder_path,
                             peft_model_name,
                             true /*trainable*/,
                             optim_config,
                             false /*init_lora_weights*/,
                             llm_model_name,
                             use_full_precision ? "fp32" : "fp16");

  GenerationConfig generationConfig(do_sample, temperature, topp);
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_verbose(verbose);
  rm->set_max_requests_per_batch(
      max_requests_per_batch +
      (int)enable_peft_finetuning); // add one slot for finetuning if needed
  // rm->set_max_concurrent_adapters(max_requests_per_batch +
  //                                 (int)enable_peft_finetuning);
  rm->set_max_concurrent_adapters(1);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->register_tokenizer(
      model_type, bos_token_id, eos_token_ids, tokenizer_filepath);
  std::string output_filepath = join_path(
      {file_paths.output_folder_path, "output.log"});
  rm->register_output_filepath(output_filepath);
  rm->set_enable_peft_finetuning(enable_peft_finetuning);
  rm->set_max_finetuning_sequence_length(1024);

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
  int tot_num_layers_in_model = model.current_transformer_layer_id+1;
  rm->set_num_transformer_layers(tot_num_layers_in_model);
  // if (num_layers_per_finetuning_step > 0) {
  //   rm->set_num_layers_per_finetuning_step(num_layers_per_finetuning_step);
  // }

  // Start background server
  rm->start_background_server(&model);

  
  PEFTModelID *peft_model_id_finetuning = model.register_peft_adapter(peft_config_finetuning);

  // Run workload
  {
    std::cout << "----------warmup started--------------" << std::endl;
    std::vector<Request> warmup_requests = make_warmup_requests(10, 1000, peft_model_id_finetuning);
    std::vector<GenerationResult> warmup_result = model.generate(warmup_requests);
    rm->set_inference_finished(false); // reset inference finished flag
    std::cout << "----------warmup finished--------------" << std::endl << std::endl << std::endl;

    for (int max_fwd_tokens_per_batch : max_fwd_finetuning_tokens) {
      rm->set_max_fwd_finetuning_tokens_per_batch(max_fwd_tokens_per_batch);
      for (int num_bwd_layers_per_step : num_layers_per_finetuning_step) {
        rm->set_num_layers_per_finetuning_step(num_bwd_layers_per_step);
        std::cout << "Benchmarking overhead of " << max_fwd_tokens_per_batch << " fwd tokens and " << num_bwd_layers_per_step << " bwd layers per step." 
        << " Run idx: " << rm->run_idx << std::endl;
        std::vector<Request> requests = make_requests(max_requests_per_batch, 
                                                      1024, 
                                                      max_fwd_tokens_per_batch, 
                                                      tot_num_layers_in_model,
                                                      num_bwd_layers_per_step,
                                                      peft_model_id_finetuning);
        std::vector<GenerationResult> result = model.generate(requests);
        std::cout << "----------inference finished--------------" << std::endl << std::endl << std::endl;
      }
    }
  }

  // terminate the request manager by stopping the background thread
  rm->terminate_background_server();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }
  std::string dataset_name = "overhead_test";
  std::cout << "Saving profiling info..." << std::endl;
  rm->save_profiling_info_to_csv(file_paths.output_folder_path,
                                  dataset_name,
                                 llm_model_name,
                                 ffconfig.tensor_parallelism_degree,
                                 max_requests_per_batch,
                                 max_tokens_per_batch,
                                 0.0, // arrival rate
                                 10); // num_warmup_requests

  if (peft_model_id_finetuning != nullptr) {
    free(peft_model_id_finetuning);
  }

  std::cout << "----------inference finished--------------" << std::endl;

  // free tokenizer space in memory
}

void FlexFlow::register_custom_tasks() {}
