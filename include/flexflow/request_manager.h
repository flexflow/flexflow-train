/* Copyright 2023 CMU, Stanford, Facebook, LANL
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

#pragma once

#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/utils/file_loader.h"
#include <future>
#include <mutex>
#include <tokenizers_cpp.h>

namespace FlexFlow {

class FFModel;
class BeamTree;
class RequestManager;
using tokenizers::Tokenizer;
using RequestGuid = BatchConfig::RequestGuid;

class InferenceManager {
public:
  InferenceManager();
  static InferenceManager *get_inference_manager();
  void compile_model_and_allocate_buffer(FFModel *model);
  void init_operators_inference(FFModel *model);
  InferenceResultFuture
      inference(FFModel *model, int index, BatchConfig const &bc);
  InferenceResultFuture
      inference(FFModel *model, int index, BatchConfigFuture const &bc);
  std::vector<FinetuningBwdFuture>
      peft_bwd(FFModel *model, int index, BatchConfigFuture const &bc);
  void load_input_tokens_from_batch_config(FFModel *model,
                                           BatchConfigFuture const &bc,
                                           ParallelTensor const input,
                                           FFHandler *handlers);
  void load_positions(FFModel *model,
                      BatchConfigFuture const &bc,
                      ParallelTensor position_input,
                      int offset);
  void register_model_weights_loader(FFModel *, FileDataLoader *);
  void load_inference_metadata_batch_config(FFModel *model,
                                            BatchConfigFuture const &bc,
                                            FFHandler *handlers);

public:
  std::unordered_map<ParallelTensor, std::vector<ParallelTensor>> tensor_buffer;
  std::unordered_map<FFModel *, FileDataLoader *> model_weights_loaders;
};

#define REQ_RECEIVED_STEP_IDX -2
#define REQ_START_TIME_STEP_IDX -1

struct StepProfileInfo {
  int step_idx;
  int run_idx;
  bool is_warmup_step;
  int num_inference_requests;
  int num_prefilling_tokens;
  int num_decoding_tokens;
  int num_finetuning_fwd_tokens;
  int num_finetuning_bwd_tokens;
  int num_bwd_layers;
  long long timestamp;
};

struct InferenceReqProfileInfo {
  RequestGuid request_guid;
  int decoding_step_idx;
  long long timestamp;
};


struct Request {
  enum Status {
    PENDING = 101,   // loading prompt
    RUNNING = 102,   // running inference
    COMPLETED = 103, // finished and verified
    FINISHING = 104, // finishing request, but not yet verified
  };
  enum FinetuningStatus {
    FORWARD_PHASE = 201,
    BACKWARD_PHASE = 202,
  };
  struct PeftFinetuningInfo {
    FinetuningStatus status = FORWARD_PHASE;
    std::string dataset_filepath;
    int max_training_steps = 1;
    // overall state
    int completed_training_steps = 0;
    // fwd state
    int dataset_entry_processed_tokens = 0;
    std::vector<float> finetuning_losses;
    // bwd state
    int last_processed_bwd_layer = INT_MAX;
    // how many gradient accumulation steps to do before updating the weights.
    // if left as -1, it will be set to the number of entries in the dataset
    int gradient_accumulation_steps = -1;
    // std::vector<int> finetuning_tokens_per_batch;
  };
  RequestType req_type = REQ_INFERENCE;
  RequestGuid guid = BatchConfig::INVALID_GUID;
  int max_length = -1;
  int max_new_tokens = -1;
  int benchmarking_tokens = -1;
  bool add_special_tokens = true;
  bool warmup = false;
  Status status = PENDING;
  long long arrival_time_us = 0;
  // inference fields
  std::string prompt;
  std::vector<BatchConfig::TokenId> tokens;

  // peft fields
  PEFTModelID peft_model_id = PEFTModelID::NO_ID;
  PeftFinetuningInfo peft_finetuning_info;
  std::vector<std::vector<BatchConfig::TokenId>> dataset;

  // speculation fields
  int initial_len = 0;
  int ssm_cache_size = 0;
  int llm_cache_size = 0;
  std::vector<struct BeamTree> beam_trees;

  Request() = default;
  static Request from_other(Request const &other);

  friend std::ostream &operator<<(std::ostream &os, Request const &req);
};

// store the result of beam search
struct BeamTree {
  struct treeLayer {
    BeamSearchBatchConfig::TokenId
        tokens[BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    int parent_ids[BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    float probs[BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    int nodes_num_this_layer = 0;
  };
  treeLayer treeLayers[BeamSearchBatchConfig::MAX_BEAM_DEPTH + 1];
};

// struct BeamTree_v2 {
//   std::vector<BatchConfig::TokenId> tokens;
//   std::vector<int> parent_ids;
//   std::vector<float> probs;
// };

class RequestManager {
public:
  enum Status {
    INITIALIZED = 1001,
    SERVING = 1002,
    TERMINATED = 1003,
  };
  using TokenId = BatchConfig::TokenId;

  RequestManager();
  static RequestManager *get_request_manager();
  size_t get_num_processed_requests();
  size_t get_num_ssms();

  bool load_request_token_ids(Request &request);

  void set_verbose(bool verbose);
  void set_max_requests_per_batch(int max_num_requests);
  int get_max_requests_per_batch();
  void set_max_tokens_per_batch(int max_num_tokens);
  int get_max_tokens_per_batch();
  void set_max_fwd_finetuning_tokens_per_batch(int max_num_tokens);
  int get_max_fwd_finetuning_tokens_per_batch();
  void set_max_spec_tree_token_num(int max_num_tokens);
  int get_max_spec_tree_token_num();
  int get_max_verify_tokens_per_batch();
  int get_max_sequence_length();
  void set_max_sequence_length(int max_seq_length);
  int get_max_finetuning_sequence_length();
  void set_max_finetuning_sequence_length(int max_seq_length);
  void push_spec_infer_tree_width(int tree_width);
  void set_enable_peft_finetuning(bool enable_peft_finetuning_);
  void set_inference_finished(bool finished = true);
  int register_ssm_model(FFModel *model);
  void register_tokenizer(ModelType model_type,
                          int bos_token_id,
                          std::vector<int> eos_token_ids,
                          std::string const &path);
  void register_output_filepath(std::string const &);
  void set_peft_config(PEFTModelID const &peft_model_id,
                       LoraLinearConfig const &peft_config);
  LoraLinearConfig const &get_peft_config(PEFTModelID const &peft_model_id);
  void set_max_lora_rank(int max_lora_rank);
  void set_max_concurrent_adapters(int max_concurrent_adapters);
  int get_max_lora_rank();
  int get_max_concurrent_adapters();
  void set_num_transformer_layers(int num_transformer_layers);
  int get_num_transformer_layers();
  void set_num_layers_per_finetuning_step(int num_layers_per_finetuning_step);
  int get_num_layers_per_finetuning_step();
  void initBitMask(BatchConfig::BitMask &bitmask, int initLength);
  void appendPendingRequest(BatchConfig::BitMask &bitmask, int initLength);
  void appendBitMask(BatchConfig::BitMask &bitmask,
                     int newNodes,
                     int preBeamSize,
                     int old_sub_num,
                     BeamTree const tree,
                     int currentDepth);
  void updateBitMask(BatchConfig::BitMask &bitmask,
                     int initLength,
                     int non_tree_size);

  FFModel *get_ssm_model(int model_id);

  void serve_incr_decoding(FFModel *model);
  void serve_spec_infer(FFModel *model);
  GenerationResult get_generation_result(RequestGuid const &guid);
  RequestGuid assign_next_guid();
  RequestGuid register_new_request(Request const &request_);
  RequestGuid register_new_peft_request(Request const &request_);

  // Methods to start and terminate request manager's background task
  void start_background_server(FFModel *model);
  bool is_background_server_terminated();
  void terminate_background_server();
  static void terminate_background_server_at_exit();
  // Methods to check and mark request completion
  bool is_request_completed(RequestGuid const &guid);
  void trigger_request_completion_future(RequestGuid const &guid);
  // Methods for preparing next batches
  bool is_eos_token(int token_id);
  bool inf_req_completed(BatchConfig const &old_bc, int i);
  void check_batch(BatchConfig const &old_bc, BatchConfig const &new_bc);
  void add_peft_config_to_request_info(BatchConfig &bc,
                                       int req_idx,
                                       LoraLinearConfig const &peft_config);

  // helpers for prepare_next_batch
  void process_inf_req_progress(BatchConfig const &old_fwd_bc,
                                InferenceResult const &result);
  void handle_completed_inf_req(BatchConfig const &old_bc, int i);
  void add_continuing_inf_req_to_new_batch(BatchConfig &new_bc,
                                           BatchConfig const &old_bc,
                                           int &num_active_req,
                                           int &num_concurrent_inf_adapters,
                                           int i);
  void add_new_inf_req(BatchConfig &new_bc,
                       int &num_active_req,
                       int &num_concurrent_inf_adapters,
                       int i);
  void handle_completed_finetuning_req(BatchConfig const &old_finetuning_bc);
  void add_finetuning_req_fwd_batch(BatchConfig &new_bc);
  void add_finetuning_req_bwd_batch(BatchConfig &new_bc);
  bool finetuning_fwd_work_available();
  bool finetuning_bwd_work_available();
  void process_finetuning_req_fwd_progress(BatchConfig const &old_bc,
                                           InferenceResult const &result);
  void process_finetuning_req_bwd_progress(BatchConfig const &old_bc);
  void process_work_from_old_batch(BatchConfig const &old_bc,
                                     InferenceResult const &result);
  void record_decoding_req_profiling_info(BatchConfig const &old_fwd_bc, int req_idx);
  void record_step_profile_info(BatchConfig const &old_bc);
  void save_profiling_info_to_csv(std::string output_folder,
                                  std::string dataset_name,
                                  std::string llm_model_name,
                                  int tensor_parallelism_degree,
                                  int max_requests_per_batch,
                                  int max_tokens_per_batch,
                                  double arrival_rate,
                                  int num_warmup_requests);
  BatchConfig prepare_next_bwd_batch(BatchConfig &new_bc);
  BatchConfig prepare_next_fwd_batch(BatchConfig const &old_bc,
                                     InferenceResult const &result);
  BatchConfigFuture prepare_next_batch(BatchConfigFuture const &old_bc,
                                       InferenceResultFuture const &result,
                                       std::vector<FinetuningBwdFuture> const &bwd_f,
                                       Legion::Context ctx,
                                       Legion::Runtime *runtime);
  BeamSearchBatchConfig
      prepare_next_batch_beam(BeamSearchBatchConfig const &old_bc,
                              BeamInferenceResult const &result);
  BeamSearchBatchConfigFuture
      prepare_next_batch_beam(BeamSearchBatchConfigFuture const &old_bc,
                              BeamInferenceResultFuture const &result,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
  BeamSearchBatchConfig
      prepare_next_batch_init(TreeVerifyBatchConfig const &old_bc,
                              InferenceResult const &result,
                              int model_id);
  BeamSearchBatchConfigFuture
      prepare_next_batch_init(TreeVerifyBatchConfigFuture const &old_bc,
                              InferenceResultFuture const &result,
                              int model_id,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
  TreeVerifyBatchConfig prepare_next_batch_verify(
      std::vector<BeamSearchBatchConfig> const &old_batches);
  TreeVerifyBatchConfigFuture prepare_next_batch_verify(
      std::vector<BeamSearchBatchConfigFuture> const &old_batches,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  void store_beam_metadata(BeamSearchBatchConfig const &old_bc,
                           BeamInferenceResult const &result);
  void update_beam_metadata(BeamSearchBatchConfig &new_bc,
                            BeamSearchBatchConfig const &old_bc,
                            BeamTree &tree,
                            int request_index);

  std::vector<std::pair<BatchConfig::TokenId, int>>
      traverse_beam_tree(BeamSearchBatchConfig const &old_bc,
                         int request_index,
                         int first_token_depth_in_request);

  // remove guid after put the cached tree in request
  std::vector<std::pair<BatchConfig::TokenId, int>> merge_dfs_trees(
      std::vector<std::vector<std::pair<BatchConfig::TokenId, int>>>
          input_trees,
      int root_depth,
      RequestGuid guid);

  std::vector<std::pair<BatchConfig::TokenId, int>> traverse_verify_tree(
      size_t guid,
      std::vector<std::pair<BatchConfig::TokenId, int>> const
          &inputSerializedTree,
      std::vector<std::pair<BatchConfig::TokenId, int>> const
          &outputSerializedTree);
  static void background_serving_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  static void
      load_tokens_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      load_positions_task(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);

  static void
      load_batch_config_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static BatchConfig prepare_next_batch_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static BeamSearchBatchConfig prepare_next_batch_beam_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static BeamSearchBatchConfig prepare_next_batch_init_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static TreeVerifyBatchConfig prepare_next_batch_verify_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  
  int run_idx=0;

private:
  // configuration parameters
  int max_requests_per_batch;
  int max_tokens_per_batch;
  int max_fwd_finetuning_tokens_per_batch;
  int max_spec_tree_token_num;
  int max_sequence_length;
  int max_finetuning_sequence_length;
  Status request_manager_status;

  // peft
  std::unordered_map<PEFTModelID, LoraLinearConfig> peft_configs;
  int max_lora_rank = 32;
  int max_concurrent_adapters = 0;
  // peft benchmarking
  bool enable_peft_finetuning = false;
  bool inference_finished = false;
  int num_transformer_layers = 0;
  int num_layers_per_finetuning_step = 0;

  // tree width in each speculative step, if not specified 1
  std::vector<int> spec_infer_tree_width;

  // private fields
  std::unique_ptr<Tokenizer> tokenizer_;
  bool verbose;
  ModelType model_type;
  int bos_token_id;
  std::vector<int> eos_token_ids;
  bool old_llama_tokenizer = false;
  std::string output_filepath;
  std::queue<Request> pending_infr_request_queue;
  std::queue<Request> pending_peft_request_queue;
  std::unordered_map<RequestGuid, Request> all_requests;
  std::unordered_map<RequestGuid, GenerationResult> request_generation_results;
  std::mutex request_queue_mutex;
  std::unordered_map<RequestGuid, std::promise<void> *> request_to_promise;
  std::mutex request_to_promise_mutex;
  RequestGuid next_available_guid;

  // TODO: Move this two vector to request struct
  std::unordered_map<RequestGuid,
                     std::vector<std::pair<BatchConfig::TokenId, int>>>
      dfs_tree_inputs;
  std::unordered_map<RequestGuid, std::vector<std::pair<int, int>>>
      committed_tokens;

  // Multi-model support
  std::vector<FFModel *> ssm_models;

  // Performance profiling
  size_t num_processed_requests;

  // Background server handler
  Legion::Future background_server_handler;

  std::vector<StepProfileInfo> step_profile_infos;
  std::vector<InferenceReqProfileInfo> inf_req_profile_infos;
  int step_idx=0;

private:
  struct ProfileInfo {
    int llm_decoding_steps;
    int ssm_decoding_steps;
    double start_time, finish_time;
    double registration_time, first_token_time;
    bool first_token_time_set = false;
  };
  std::unordered_map<RequestGuid, ProfileInfo> profiling_requests;
  double total_request_run_time;
};

}; // namespace FlexFlow
