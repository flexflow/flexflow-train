#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PARALLEL_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PARALLEL_TENSOR_BACKING_H

#include "kernels/allocation.h"
#include "local-execution/tensor_slot_backing.dtg.h"
#include "local-pcg-execution/local_parallel_tensor_backing.dtg.h"
#include "local-pcg-execution/parallel_tensor_accessors_w.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "task-spec/task_binding.h"
#include "task-spec/training_tensor_slot_id_t.dtg.h"
#include "task-spec/training_parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

LocalParallelTensorBacking construct_local_parallel_tensor_backing(
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorShape> const &training_ptensor_shapes,
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorAccessorsW> const &preallocated_ptensors,
    Allocator &);

ParallelTensorAccessorsW get_accessors_for_training_ptensor(LocalParallelTensorBacking const &,
                                                            training_parallel_tensor_guid_t);

LocalTensorBacking
  get_local_tensor_backing_for_device(LocalParallelTensorBacking const &backing,
                                      MachineSpaceCoordinate const &);

std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
  construct_tensor_slots_backing_for_binding_and_task(LocalParallelTensorBacking const &backing,
                                                      TaskBinding const &binding,
                                                      MachineSpaceCoordinate const &task_device);

} // namespace FlexFlow

#endif
