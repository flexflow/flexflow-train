
#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "local-execution/allocated_tensors.dtg.h"
#include "local-execution/gradient_tensor_source.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "local-execution/loss_tensor_source.h"
#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/unallocated_tensors.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/lowered_tensor_t.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/tensor_role.dtg.h"

namespace FlexFlow {

GenericTensorAccessorW get_tensor(LocalTensorBacking const &,
                                  TensorTypeVariant const &);

std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
    get_tensor_backings(
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const &,
        std::unordered_map<TensorTypeVariant, TensorShape> const &,
        Allocator &);

std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
    merge_optimizer_mappings(
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
            &allocated,
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
            &unallocated);

LocalTensorBacking construct_local_tensor_backing(AllocatedTensors const &,
                                                  UnallocatedTensors const &,
                                                  Allocator &);

TensorSlotsBacking construct_tensor_slots_backing(LocalTensorBacking const &,
                                                  TaskBinding const &);

} // namespace FlexFlow

#endif
