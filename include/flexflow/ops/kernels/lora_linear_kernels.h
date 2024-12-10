#ifndef _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/lora_linear.h"
#include "flexflow/utils/peft_weight_allocator.h"

namespace FlexFlow {

using Legion::Context;
using Legion::Runtime;

class LoraLinearMeta : public OpMeta {
public:
  LoraLinearMeta(FFHandler handle, LoraLinear const *li);
  ~LoraLinearMeta(void);
  PEFTMemoryManager *peft_memory_manager;
};

namespace Kernels {
namespace LoraLinear {

bool lora_applies_to_this_layer(LoraLinearMeta *m,
                                LoraLinearConfig const &config);

// void init_kernel_wrapper(LoraLinearMeta *m, int seed);
void inference_kernel_wrapper(LoraLinearMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output);
void peft_bwd_kernel_wrapper(Context ctx,
                             Runtime *runtime,
                             LoraLinearMeta *m,
                             BatchConfig const *bc,
                             int shard_id,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);
void save_peft_weights_if_needed(LoraLinearMeta *m,
                                 BatchConfig const *bc,
                                 int in_dim,
                                 int out_dim,
                                 int shard_id);

namespace Internal {
// template <typename DT>
// void init_kernel(LoraLinearMeta *m, int seed, ffStream_t stream);
template <typename DT>
void inference_kernel(LoraLinearMeta *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int in_dim,
                      int out_dim,
                      ffStream_t stream);
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
                     ffStream_t stream);
} // namespace Internal
} // namespace LoraLinear
} // namespace Kernels
} // namespace FlexFlow
#endif // _FLEXFLOW_OPS_KERNELS_LORA_LINEAR_KERNELS_H
