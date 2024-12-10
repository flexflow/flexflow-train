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

#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/lora_linear_kernels.h"
#include "flexflow/utils/cuda_helper.h"
#include <random>
#include <vector>

namespace FlexFlow {

LoraLinearMeta::LoraLinearMeta(FFHandler handler, LoraLinear const *li)
    : OpMeta(handler, li) {}

LoraLinearMeta::~LoraLinearMeta(void) {}

std::string
    get_peft_dbg_folder(LoraLinearMeta const *m, int shard_id, bool is_fwd) {
  std::string op_name_without_uid = LoraLinear::get_op_name_without_uid(m);
  fs::path dst_filepath;
  if (is_fwd) {
    dst_filepath = get_dst_folder("fwd", m->decoding_step, shard_id);
  } else {
    dst_filepath = get_dst_folder("bwd", m->bwd_step, shard_id);
  }
  if (m->layer_guid.model_id > 0) {
    assert(false && "Model ID > 0 not supported yet");
  }
  std::string layername = "layers." +
                          std::to_string(m->layer_guid.transformer_layer_id) +
                          "." + op_name_without_uid;
  dst_filepath /= layername;
  return dst_filepath.string();
}

namespace Kernels {
namespace LoraLinear {

void inference_kernel_wrapper(LoraLinearMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;

  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    Internal::inference_kernel<float>(m,
                                      bc,
                                      input.get_float_ptr(),
                                      output.get_float_ptr(),
                                      in_dim,
                                      out_dim,
                                      stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::inference_kernel<half>(m,
                                     bc,
                                     input.get_half_ptr(),
                                     output.get_half_ptr(),
                                     in_dim,
                                     out_dim,
                                     stream);
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [LoraLinear] forward time = %.2lfms\n", m->op_name, elapsed);
    // print_tensor<float>((float*)input_ptr, in_dim * batch_size,
    // "[LoraLinear:forward:input]"); print_tensor<float>((float*)weight_ptr,
    // in_dim
    // * out_dim, "[LoraLinear:forward:kernel]");
    // print_tensor<float>((float*)output_ptr, out_dim * batch_size,
    // "[LoraLinear:forward:output]");
  }
}

void peft_bwd_kernel_wrapper(Context ctx,
                             Runtime *runtime,
                             LoraLinearMeta *m,
                             BatchConfig const *bc,
                             int shard_id,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  int in_dim = input_grad.domain.hi()[0] - input_grad.domain.lo()[0] + 1;
  int out_dim = output_grad.domain.hi()[0] - output_grad.domain.lo()[0] + 1;
  if (m->input_type[0] == DT_FLOAT) {
    Internal::peft_bwd_kernel<float>(ctx,
                                     runtime,
                                     m,
                                     bc,
                                     shard_id,
                                     input_grad.get_float_ptr(),
                                     output_grad.get_float_ptr(),
                                     in_dim,
                                     out_dim,
                                     stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::peft_bwd_kernel<half>(ctx,
                                    runtime,
                                    m,
                                    bc,
                                    shard_id,
                                    input_grad.get_half_ptr(),
                                    output_grad.get_half_ptr(),
                                    in_dim,
                                    out_dim,
                                    stream);
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [LoraLinear] PEFT Bwd time = %.2lfms\n", m->op_name, elapsed);
    // print_tensor<float>((float*)input_ptr, in_dim * batch_size,
    // "[LoraLinear:forward:input]"); print_tensor<float>((float*)weight_ptr,
    // in_dim
    // * out_dim, "[LoraLinear:forward:kernel]");
    // print_tensor<float>((float*)output_ptr, out_dim * batch_size,
    // "[LoraLinear:forward:output]");
  }
}

bool lora_applies_to_this_layer(LoraLinearMeta *m,
                                LoraLinearConfig const &config) {
  for (std::string s : config.target_modules) {
    std::string n(m->op_name);
    if (n.find(s) != std::string::npos) {
      return true;
    }
  }
  return false;
}

namespace Internal {

template <typename DT>
void inference_kernel(LoraLinearMeta *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int in_dim,
                      int out_dim,
                      ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->input_type[1]);
  cudaDataType_t lr_actv_type = output_type;
  assert(input_type == output_type);
  cudaDataType_t weight_type = output_type;
  cudaDataType_t compute_type = output_type;

  int num_peft_requests = 0;
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i] ||
        bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
      continue;
    }
    if (bc->requestsInfo[i].finetuning_request) {
      num_peft_requests++;
    }
    std::string peft_model_config_str =
        std::string(bc->requestsInfo[i].peft_model_config_str);
    LoraLinearConfig lora_config =
        LoraLinearConfig::deserialize_from_json_string(peft_model_config_str);
    if (!lora_applies_to_this_layer(m, lora_config)) {
      continue;
    }
    // std::cout << "Lora layer activated!" << std::endl;
    // std::cout << "Lora Config: " << peft_model_config_str << std::endl;
    assert(lora_config.trainable == bc->requestsInfo[i].finetuning_request &&
           "Trainable flag mismatch");
    int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    // assert(num_peft_tokens == bc->num_finetuning_fwd_tokens());
    // int max_peft_tokens = bc->requestsInfo[i].max_length;
    int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
    LoraLinearWeight weight = m->peft_memory_manager->get_peft(
        bc->requestsInfo[i].peft_model_id, lora_config);
    void *intermediate_result_ptr = (bc->requestsInfo[i].finetuning_request)
                                        ? weight.low_rank_activation
                                        : m->handle.workSpace;
    if (bc->requestsInfo[i].finetuning_request) {
      checkCUDA(cudaMemcpyAsync(weight.input_activation,
                                input_ptr + first_token_offset * in_dim,
                                data_type_size(m->input_type[0]) *
                                    num_peft_tokens * in_dim,
                                cudaMemcpyDeviceToDevice,
                                stream));
    } else {
      // use workspace to save intermediate result
      assert(m->handle.workSpaceSize >= data_type_size(m->input_type[1]) *
                                            num_peft_tokens * lora_config.rank);
    }
    DT alpha = 1.0f, beta = 0.0f;
    // buffer = weight_first * input
    // [rank, num_peft_tokens] = [in_dim, rank].T * [in_dim, num_peft_tokens]
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           lora_config.rank,
                           num_peft_tokens,
                           in_dim,
                           &alpha,
                           weight.w0_ptr,
                           weight_type,
                           in_dim,
                           input_ptr + first_token_offset * in_dim,
                           input_type,
                           in_dim,
                           &beta,
                           intermediate_result_ptr,
                           lr_actv_type,
                           lora_config.rank,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // output = weight_second * buffer
    // [out_dim, num_peft_tokens] = [rank, out_dim].T * [rank, num_peft_tokens]
    // Note that we use alpha in both places since we do
    // an in-place update for LoraLinear
    DT scaling_constant = (DT)(lora_config.lora_alpha / lora_config.rank);
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           out_dim,
                           num_peft_tokens,
                           lora_config.rank,
                           &scaling_constant,
                           weight.w1_ptr,
                           weight_type,
                           lora_config.rank,
                           intermediate_result_ptr,
                           lr_actv_type,
                           lora_config.rank,
                           &alpha,
                           output_ptr + first_token_offset * out_dim,
                           output_type,
                           out_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  assert(num_peft_requests <= 1);
}

template <typename DT>
__global__ void sgd_update(size_t count,
                           float lr,
                           float weight_decay,
                           float momentum,
                           bool nesterov,
                           DT const *WGrad,
                           DT *V,
                           DT *W) {
  // Refernce https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
  CUDA_KERNEL_LOOP(i, count) {
    DT gt = WGrad[i] + (DT)weight_decay * W[i];
    if (momentum > 0.0f) {
      V[i] = V[i] * (DT)momentum + gt;
      if (nesterov) {
        gt = gt + (DT)momentum * V[i];
      } else {
        gt = V[i];
      }
    }
    W[i] -= (DT)lr * gt;
  }
}

template <typename DT>
void peft_bwd_kernel(Context ctx,
                     Runtime *runtime,
                     LoraLinearMeta *m,
                     BatchConfig const *bc,
                     int shard_id,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     int in_dim,
                     int out_dim,
                     ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type[0]);
  assert(input_type == output_type);
  cudaDataType_t weight_type = output_type;
  cudaDataType_t lr_actv_type = output_type;
  cudaDataType_t compute_type = output_type;

  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();
  std::string peft_model_config_str =
      std::string(bc->requestsInfo[i].peft_model_config_str);
  LoraLinearConfig lora_config =
      LoraLinearConfig::deserialize_from_json_string(peft_model_config_str);
  if (!lora_applies_to_this_layer(m, lora_config)) {
    return;
  }
  // std::cout << "Lora layer activated!" << std::endl;
  // std::cout << "Lora Config: " << peft_model_config_str << std::endl;
  assert(lora_config.trainable == bc->requestsInfo[i].finetuning_request &&
         "Trainable flag mismatch");
  m->peft_memory_manager->check_ft_model_id(bc->requestsInfo[i].peft_model_id);
  int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
  assert(num_peft_tokens == bc->num_finetuning_bwd_tokens());
  // int max_peft_tokens = bc->requestsInfo[i].max_length;
  // int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
  LoraLinearWeight weight = m->peft_memory_manager->get_peft(
      bc->requestsInfo[i].peft_model_id, lora_config);
  DT scaling_constant = (DT)(lora_config.lora_alpha / lora_config.rank);

  // Compute LORA_B weight's gradient
  if (bc->requestsInfo[i].optimizer_tasks.compute_gradients) {
    DT alpha = 1.0f;
    DT beta = (bc->requestsInfo[i].optimizer_tasks.reset_gradients_to_zero)
                  ? 0.0f
                  : 1.0f;
    // std::cout << "Lora B gradient computation, beta = " << (float) beta <<
    // std::endl;
    if (m->inference_debugging) {
      // save result to file for checking
      std::string filename =
          get_peft_dbg_folder(m, shard_id, false) + ".low_rank_activation";
      std::cout << "Save low_rank_activation (" << lora_config.rank << ", "
                << num_peft_tokens << ") to " << filename << std::endl;
      save_tensor(static_cast<const DT *>(weight.low_rank_activation),
                  lora_config.rank * num_peft_tokens,
                  filename.c_str());
    }
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           lora_config.rank,
                           out_dim,
                           num_peft_tokens,
                           &scaling_constant,
                           weight.low_rank_activation,
                           lr_actv_type,
                           lora_config.rank,
                           output_grad_ptr,
                           output_type,
                           out_dim,
                           &beta,
                           weight.w1_grad_ptr,
                           weight_type,
                           lora_config.rank,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  // Compute LORA_B input's (and LORA_A output's) gradient inplace in
  // low_rank_activation
  {
    DT alpha = 1.0f, beta = 0.0f;
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           lora_config.rank,
                           num_peft_tokens,
                           out_dim,
                           &scaling_constant,
                           weight.w1_ptr,
                           weight_type,
                           lora_config.rank,
                           output_grad_ptr,
                           output_type,
                           out_dim,
                           &beta,
                           weight.low_rank_activation,
                           lr_actv_type,
                           lora_config.rank,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  // Compute LORA_A weight's gradient
  if (bc->requestsInfo[i].optimizer_tasks.compute_gradients) {
    DT alpha = 1.0f;
    DT beta = (bc->requestsInfo[i].optimizer_tasks.reset_gradients_to_zero)
                  ? 0.0f
                  : 1.0f;
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           in_dim,
                           lora_config.rank,
                           num_peft_tokens,
                           &alpha,
                           weight.input_activation,
                           input_type,
                           in_dim,
                           weight.low_rank_activation,
                           lr_actv_type,
                           lora_config.rank,
                           &beta,
                           weight.w0_grad_ptr,
                           weight_type,
                           in_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  // Compute input gradient
  // NOTE: we use beta=1 for input_grad to accumulate gradients when needed
  if (input_grad_ptr != nullptr) {
    DT alpha = 1.0f;
    DT beta = m->reset_input_grads[0] ? 0.0f : 1.0f;
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           in_dim,
                           num_peft_tokens,
                           lora_config.rank,
                           &alpha,
                           weight.w0_ptr,
                           weight_type,
                           in_dim,
                           weight.low_rank_activation,
                           lr_actv_type,
                           lora_config.rank,
                           &beta,
                           input_grad_ptr,
                           input_type,
                           in_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  if (bc->requestsInfo[i].optimizer_tasks.update_weights) {
    assert(lora_config.optimizer_config != nullptr);
    int w0_num_elements = lora_config.rank * in_dim;
    int w1_num_elements = lora_config.rank * out_dim;

    // Get optimizer config

    if (lora_config.optimizer_config->getType() == "SGD") {
      LoraSGDOptimizerConfig const *sgd_config =
          static_cast<LoraSGDOptimizerConfig const *>(
              lora_config.optimizer_config);
      // LoRA_A weight is split in tensor parallelism, so no need to apply
      // all-reduce
      sgd_update<<<GET_BLOCKS(w0_num_elements), CUDA_NUM_THREADS, 0, stream>>>(
          w0_num_elements,
          sgd_config->lr,
          sgd_config->weight_decay,
          sgd_config->momentum,
          sgd_config->nesterov,
          static_cast<DT const *>(weight.w0_grad_ptr),
          static_cast<DT *>(weight.w0_v_values_ptr),
          static_cast<DT *>(weight.w0_ptr));
      // LoRA_B weight is replicated w tensor parallelism, so we need to sync
      // and sum first
#ifdef FF_USE_NCCL
      ncclDataType_t nccl_data_type = ff_to_nccl_datatype(m->output_type[0]);
      runtime->concurrent_task_barrier(ctx);
      checkNCCL(ncclAllReduce(static_cast<DT const *>(weight.w1_grad_ptr),
                              static_cast<DT *>(weight.w1_grad_ptr),
                              w1_num_elements,
                              nccl_data_type,
                              ncclSum,
                              m->handle.ncclComm,
                              stream));
      runtime->concurrent_task_barrier(ctx);
#else
      assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
      sgd_update<<<GET_BLOCKS(w1_num_elements), CUDA_NUM_THREADS, 0, stream>>>(
          w1_num_elements,
          sgd_config->lr,
          sgd_config->weight_decay,
          sgd_config->momentum,
          sgd_config->nesterov,
          static_cast<DT const *>(weight.w1_grad_ptr),
          static_cast<DT *>(weight.w1_v_values_ptr),
          static_cast<DT *>(weight.w1_ptr));
    } else if (lora_config.optimizer_config->getType() == "Adam") {
      assert(false && "Adam optimizer type not implemented yet");
    } else {
      assert(false && "Unsupported optimizer type");
    }
  }
}

} // namespace Internal
} // namespace LoraLinear
} // namespace Kernels
} // namespace FlexFlow
