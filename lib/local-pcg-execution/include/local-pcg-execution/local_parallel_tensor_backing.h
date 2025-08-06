#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PARALLEL_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PARALLEL_TENSOR_BACKING_H

#include "kernels/allocation.h"
#include "local-pcg-execution/local_parallel_tensor_backing.dtg.h"
#include "local-pcg-execution/parallel_tensor_accessors_w.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "task-spec/training_parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

LocalParallelTensorBacking construct_local_parallel_tensor_backing(
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorShape> const &training_ptensor_shapes,
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorAccessorsW> const &preallocated_ptensors,
    Allocator &);

ParallelTensorAccessorsW get_accessors_for_training_ptensor(LocalParallelTensorBacking const &,
                                                            training_parallel_tensor_guid_t);

} // namespace FlexFlow

#endif
