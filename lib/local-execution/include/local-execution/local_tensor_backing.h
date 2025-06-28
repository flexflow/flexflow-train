#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "local-execution/tensor_slot_backing.dtg.h"
#include "task-spec/task_binding.h"
#include "task-spec/training_computation_graph.dtg.h"
#include "task-spec/training_tensor_guid_t.dtg.h"

namespace FlexFlow {

LocalTensorBacking construct_local_tensor_backing(
    std::unordered_map<training_tensor_guid_t, TensorShape> const
        &training_tensor_shapes,
    std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated_tensors,
    Allocator &);

GenericTensorAccessorW
    get_accessor_for_training_tensor(LocalTensorBacking const &,
                                     training_tensor_guid_t);

std::unordered_map<tensor_sub_slot_id_t, TensorSlotBacking>
    construct_tensor_slots_backing_for_binding(LocalTensorBacking const &,
                                               TaskBinding const &);

} // namespace FlexFlow

#endif
