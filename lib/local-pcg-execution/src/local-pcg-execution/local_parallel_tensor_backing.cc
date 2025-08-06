#include "local-pcg-execution/local_parallel_tensor_backing.h"

namespace FlexFlow {

LocalParallelTensorBacking construct_local_parallel_tensor_backing(
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorShape> const &training_ptensor_shapes,
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorAccessorsW> const &preallocated_ptensors,
    Allocator &) {
  
  NOT_IMPLEMENTED();
}

ParallelTensorAccessorsW get_accessors_for_training_ptensor(LocalParallelTensorBacking const &,
                                                            training_parallel_tensor_guid_t) {
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
