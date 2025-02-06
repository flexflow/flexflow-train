
#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "local-execution/gradient_tensor_source.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/loss_tensor_source.h"
#include "local-execution/lowered_tensor_source.h"
#include "local-execution/optimizer_tensor_source.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/loss_tensor_t.dtg.h"
#include "task-spec/lowered_tensor_t.dtg.h"
#include "task-spec/optimizer_tensor_t.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/tensor_role.dtg.h"
#include "task-spec/tensor_type_t.dtg.h"

namespace FlexFlow {

using TensorBackingMap =
    std::unordered_map<lowered_tensor_t, GenericTensorAccessorW>;

struct LocalTensorBacking {
  LocalTensorBacking() = default;
  LocalTensorBacking(
      std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const
          &allocated_tensor_backings,
      std::unordered_set<tensor_guid_t> const &allocated_tensor_guids,
      std::unordered_map<tensor_guid_t, gradient_tensor_t> const
          &allocated_gradient_mapping,
      std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
          &allocated_optimizer_mapping,
      std::unordered_set<loss_tensor_t> const &allocated_loss_tensors);

  lowered_tensor_t allocate_tensor(TensorShape const &, Allocator &);

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

private:
  lowered_tensor_t insert_tensor(GenericTensorAccessorW const &);
  LoweredTensorSource lowered_tensor_source;
};

void allocate_tensor_guid(LocalTensorBacking &,
                          tensor_guid_t const &,
                          TensorShape const &,
                          Allocator &);
void allocate_gradient_tensor(LocalTensorBacking &,
                              gradient_tensor_t const &,
                              tensor_guid_t const &,
                              TensorShape const &,
                              Allocator &);
void allocate_optimizer_tensors(LocalTensorBacking &,
                                std::vector<optimizer_tensor_t> const &,
                                tensor_guid_t const &,
                                TensorShape const &,
                                Allocator &);

void allocate_all_computation_graph_tensors(LocalTensorBacking &,
                                            GradientTensorSource &,
                                            ComputationGraph const &,
                                            Allocator &);
void allocate_all_optimizer_tensors(LocalTensorBacking &,
                                    OptimizerTensorSource &,
                                    ComputationGraph const &,
                                    Allocator &,
                                    OptimizerAttrs const &);
loss_tensor_t allocate_loss_tensor(LocalTensorBacking &,
                                   LossTensorSource const &,
                                   TensorShape const &,
                                   Allocator &);

TensorSlotsBacking construct_tensor_slots_backing(LocalTensorBacking const &,
                                                  TaskBinding const &);

} // namespace FlexFlow

#endif
