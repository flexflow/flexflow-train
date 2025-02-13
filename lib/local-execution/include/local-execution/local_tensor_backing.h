
#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "local-execution/allocated_tensors.dtg.h"
#include "local-execution/gradient_tensor_source.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/loss_tensor_source.h"
#include "local-execution/lowered_tensor_source.h"
#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/unallocated_tensors.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/lowered_tensor_t.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/tensor_role.dtg.h"

namespace FlexFlow {

using TensorBackingMap =
    std::unordered_map<lowered_tensor_t, GenericTensorAccessorW>;

struct LocalTensorBacking {
  LocalTensorBacking(AllocatedTensors const &,
                     UnallocatedTensors const &,
                     Allocator const &);

public:
  GenericTensorAccessorW get_tensor(TensorTypeVariant const &) const;

public:
  // tensors
  TensorBackingMap tensor_backings;

  std::unordered_map<tensor_guid_t, lowered_tensor_t> tensor_lowering_mapping;
  std::unordered_map<gradient_tensor_t, lowered_tensor_t>
      gradient_tensor_lowering_mapping;
  std::unordered_map<optimizer_tensor_t, lowered_tensor_t>
      optimizer_tensor_lowering_mapping;
  std::unordered_map<loss_tensor_t, lowered_tensor_t>
      loss_tensor_lowering_mapping;

  std::unordered_map<tensor_guid_t, gradient_tensor_t> tensor_gradient_mapping;
  std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
      tensor_optimizer_mapping;

  Allocator allocator;

private:
  lowered_tensor_t insert_tensor(TensorTypeVariant const &);
  LoweredTensorSource lowered_tensor_source;
};

UnallocatedTensors generate_unallocated_tensors(AllocatedTensors const &,
                                                std::unordered_map<tensor_guid_t, TensorAttrs> const &,
                                                GradientTensorSource &);

UnallocatedTensors
    generate_unallocated_tensors_with_optimizer(AllocatedTensors const &,
                                                std::unordered_map<tensor_guid_t, TensorAttrs> const &,
                                                GradientTensorSource &,
                                                OptimizerTensorSource &,
                                                OptimizerAttrs const &);

TensorSlotsBacking construct_tensor_slots_backing(LocalTensorBacking const &,
                                                  TaskBinding const &);

} // namespace FlexFlow

#endif
