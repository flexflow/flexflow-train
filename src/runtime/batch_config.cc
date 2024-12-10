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

#include "flexflow/batch_config.h"
#include "flexflow/request_manager.h"
#include "legion.h"
#include <cassert>
#include <climits>

namespace FlexFlow {

Legion::Logger log_bc("BatchConfig");
using Legion::Future;
using Legion::Memory;

void set_optimizer_tasks(OptimizerTasks &tasks,
                         int max_training_steps,
                         int completed_training_steps,
                         int gradient_accumulation_steps) {
  assert(max_training_steps > 0);
  assert(completed_training_steps >= 0);
  assert(gradient_accumulation_steps > 0);
  assert(completed_training_steps < max_training_steps);
  // Compute gradients should always be true
  tasks.compute_gradients = true;

  // Reset gradients to zero in the first iteration and after weight updates
  tasks.reset_gradients_to_zero =
      (completed_training_steps == 0) ||
      (completed_training_steps % gradient_accumulation_steps == 0);

  // Update weights every gradient_accumulation_steps
  tasks.update_weights =
      ((completed_training_steps + 1) % gradient_accumulation_steps == 0);

  // Save updated weights only in the very last training step
  tasks.save_updated_weights = false;
  // tasks.save_updated_weights =
  //     (completed_training_steps == max_training_steps - 1);
  if (tasks.save_updated_weights) {
    assert(tasks.update_weights);
  }
}

BatchConfig::BatchConfig() : num_tokens(0), num_generation_tokens(0) {
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    requestsInfo[i].first_token_depth_in_request = 0;
    requestsInfo[i].first_token_offset_in_batch = 0;
    requestsInfo[i].num_tokens_in_batch = 0;
    requestsInfo[i].max_length = 0;
    requestsInfo[i].request_guid = 0;
    requestsInfo[i].peft_model_id = PEFTModelID::NO_ID;
    requestsInfo[i].finetuning_request = false;
    requestsInfo[i].finetuning_backward_phase = false;
    requestsInfo[i].peft_bwd_first_layer = -1;
    requestsInfo[i].peft_bwd_last_layer = -1;
    requestsInfo[i].optimizer_tasks = {true, false, false, false};
    requestsInfo[i].prompt_phase = false;
    requestsInfo[i].batch_config_request_id = -1;
    std::memset(requestsInfo[i].peft_model_config_str, 0, MAX_PEFT_CONFIG_SIZE);
    request_completed[i] = true;
    request_running[i] = false;
  }
  for (int i = 0; i < MAX_NUM_TOKENS; i++) {
    tokensInfo[i].abs_depth_in_request = 0;
    tokensInfo[i].request_index = 0;
    tokensInfo[i].token_id = 0;
  }
}

/*static*/
BatchConfig const *BatchConfig::from_future(BatchConfigFuture const &future) {
  BatchConfig const *bc = static_cast<BatchConfig const *>(
      Future(future).get_buffer(Memory::SYSTEM_MEM));
  // Check future size
  if (bc->get_mode() == INC_DECODING_MODE) {
    assert(Future(future).get_untyped_size() == sizeof(BatchConfig));
  } else if (bc->get_mode() == BEAM_SEARCH_MODE) {
    assert(Future(future).get_untyped_size() == sizeof(BeamSearchBatchConfig));
  } else if (bc->get_mode() == TREE_VERIFY_MODE) {
    assert(Future(future).get_untyped_size() == sizeof(TreeVerifyBatchConfig));
  } else {
    assert(false && "Unsupported inference mode");
  }
  return bc;
}

InferenceMode BatchConfig::get_mode() const {
  return INC_DECODING_MODE;
}

int BatchConfig::num_active_requests() const {
  int num_requests = 0;
  for (int i = 0; i < max_requests_per_batch(); i++) {
    if (!request_completed[i]) {
      num_requests++;
    }
  }
  return num_requests;
}

int BatchConfig::num_active_tokens() const {
  return num_tokens;
}

int BatchConfig::finetuning_request_index() const {
  assert(max_requests_per_batch() > 0);
  return max_requests_per_batch() - 1;
}

int BatchConfig::num_finetuning_fwd_requests() const {
  if (request_completed[finetuning_request_index()] || 
      !requestsInfo[finetuning_request_index()].finetuning_request || 
      requestsInfo[finetuning_request_index()].finetuning_backward_phase) {
    return 0;
  }
  return 1;
}

int BatchConfig::num_finetuning_fwd_tokens() const {
  if (num_finetuning_fwd_requests() == 0) {
    return 0;
  }
  return requestsInfo[finetuning_request_index()].num_tokens_in_batch;
}

int BatchConfig::num_finetuning_bwd_requests() const {
  if (request_completed[finetuning_request_index()] || 
      !requestsInfo[finetuning_request_index()].finetuning_request || 
      !requestsInfo[finetuning_request_index()].finetuning_backward_phase) {
    return 0;
  }
  return 1;
}

int BatchConfig::num_finetuning_bwd_tokens() const {
  if (num_finetuning_bwd_requests() == 0) {
    return 0;
  }
  return requestsInfo[finetuning_request_index()].num_tokens_in_batch;
}

bool BatchConfig::peft_bwd_applies_to_this_layer(int layer) const {
  if (!requestsInfo[finetuning_request_index()].finetuning_request ||
      !requestsInfo[finetuning_request_index()].finetuning_backward_phase) {
    return false;
  }
  assert(requestsInfo[finetuning_request_index()].peft_bwd_first_layer >= 0);
  assert(requestsInfo[finetuning_request_index()].peft_bwd_last_layer >= 0);
  assert(layer >= 0);
  return (
      layer >= requestsInfo[finetuning_request_index()].peft_bwd_first_layer &&
      layer <= requestsInfo[finetuning_request_index()].peft_bwd_last_layer);
}

/*static*/
int BatchConfig::max_requests_per_batch() {
  return RequestManager::get_request_manager()->get_max_requests_per_batch();
}

/*static*/
int BatchConfig::max_tokens_per_batch() {
  return RequestManager::get_request_manager()->get_max_tokens_per_batch();
}

/*static*/
int BatchConfig::max_verify_tokens_per_batch() {
  return RequestManager::get_request_manager()
      ->get_max_verify_tokens_per_batch();
}

/*static*/
int BatchConfig::max_sequence_length() {
  return RequestManager::get_request_manager()->get_max_sequence_length();
}

/*static*/
int BatchConfig::max_finetuning_sequence_length() {
  return RequestManager::get_request_manager()->get_max_finetuning_sequence_length();
}

int BatchConfig::max_spec_tree_token_num() {
  return RequestManager::get_request_manager()->get_max_spec_tree_token_num();
}

// print InferenceResult
std::ostream &operator<<(std::ostream &os, InferenceResult const &result) {
  os << "InferenceResult {";
  os << "MAX_NUM_TOKENS: " << InferenceResult::MAX_NUM_TOKENS << ", ";
  os << "token_ids: [";
  for (int i = 0; i < 16; i++) {
    os << result.token_ids[i] << ", ";
  }
  os << "], ";
  os << "finetuning_loss: " << result.finetuning_loss;
  os << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, BatchConfig const &bc) {
  os << "@@@@@@@@@@@@@@ Batch Config (mode " << bc.get_mode()
     << ") @@@@@@@@@@@@@@" << std::endl;
  // Max values
  os << "Max number of requests: " << bc.max_requests_per_batch() << std::endl;
  os << "Max number of tokens: " << bc.max_tokens_per_batch() << std::endl;
  os << "Max sequence length: " << bc.max_sequence_length() << std::endl;
  // Current values
  os << "Number of requests: " << bc.num_active_requests() << std::endl;
  os << "Number of peft fwd requests: " << bc.num_finetuning_fwd_requests() << std::endl;
  os << "Number of peft bwd requests: " << bc.num_finetuning_bwd_requests() << std::endl;
  os << "Number of active tokens: " << bc.num_active_tokens() << std::endl;
  os << "Number of generation tokens: " << bc.num_generation_tokens
     << std::endl;
  os << "Number of peft fwd tokens: " << bc.num_finetuning_fwd_tokens() << std::endl;
  os << "Number of peft bwd tokens: " << bc.num_finetuning_bwd_tokens() << std::endl;

  // Per-request info
  os << "Per-request info:\n";
  for (int i = 0; i < bc.max_requests_per_batch(); i++) {
    if (!bc.request_completed[i]) {
      os << "  Request " << i << ":\n";
      os << "    First token depth in request: "
         << bc.requestsInfo[i].first_token_depth_in_request << std::endl;
      os << "    First token offset in batch: "
         << bc.requestsInfo[i].first_token_offset_in_batch << std::endl;
      os << "    Number of tokens in batch: "
         << bc.requestsInfo[i].num_tokens_in_batch << std::endl;
      os << "    Max sequence length: " << bc.requestsInfo[i].max_length
         << std::endl;
      os << "    BatchConfig Req ID: "
         << bc.requestsInfo[i].batch_config_request_id << std::endl;
      os << "    Prompt phase: " << bc.requestsInfo[i].prompt_phase
         << std::endl;
      os << "    GUID: " << bc.requestsInfo[i].request_guid << std::endl;
      // PEFT values
      os << "    PEFT Model ID: " << bc.requestsInfo[i].peft_model_id
         << std::endl;
      os << "    Finetuning req: " << bc.requestsInfo[i].finetuning_request
         << std::endl;
      os << "    Finetuning backward phase: "
         << bc.requestsInfo[i].finetuning_backward_phase << std::endl;
      os << "    PEFT backward first layer: "
         << bc.requestsInfo[i].peft_bwd_first_layer << std::endl;
      os << "    PEFT backward last layer: "
          << bc.requestsInfo[i].peft_bwd_last_layer << std::endl;
      os << "    optimizer_tasks: {"
         << "compute_gradients: " << std::boolalpha
         << bc.requestsInfo[i].optimizer_tasks.compute_gradients
         << ", reset_gradients_to_zero: "
         << bc.requestsInfo[i].optimizer_tasks.reset_gradients_to_zero
         << ", update_weights: "
         << bc.requestsInfo[i].optimizer_tasks.update_weights
         << ", save_updated_weights: "
         << bc.requestsInfo[i].optimizer_tasks.save_updated_weights << "}"
         << std::endl;
      os << "    Request completed: " << bc.request_completed[i] << std::endl;
      os << "    Request running: " << bc.request_running[i] << std::endl;
    }
  }

  // Per-token info
  os << "Per-token info:\n";
  for (int i = 0; i < bc.num_tokens; i++) {
    os << "  Token " << i << ":\n";
    os << "    Absolute depth in request: "
       << bc.tokensInfo[i].abs_depth_in_request << std::endl;
    os << "    Request index: " << bc.tokensInfo[i].request_index << std::endl;
    os << "    Token id: " << bc.tokensInfo[i].token_id << std::endl;
  }
  os << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  return os;
}

void BatchConfig::print() const {
  std::cout << *this << std::endl;
}

void BatchConfig::save_to_file(std::string const &filename) const {
  std::ofstream outputFile(filename);
  if (outputFile.is_open()) {
    outputFile << *this << std::endl;
    outputFile.close();
  } else {
    std::cerr << "Error: Unable to open the batch config output file: "
              << filename << std::endl;
    assert(false);
  }
}

}; // namespace FlexFlow
