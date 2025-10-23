#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_TASK_SIGNATURE_TENSOR_KEY_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_TASK_SIGNATURE_TENSOR_KEY_H

#include <unordered_set>
#include "pcg/mapped_parallel_computation_graph/task_signature_tensor_key.dtg.h"

namespace FlexFlow {

std::unordered_set<TaskSignatureTensorKey>
  all_keys_for_signature_arities(
    nonnegative_int num_inputs,
    nonnegative_int num_weights,
    nonnegative_int num_outputs);

} // namespace FlexFlow

#endif
