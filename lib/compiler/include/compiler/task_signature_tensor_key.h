#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_TASK_SIGNATURE_TENSOR_KEY_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_TASK_SIGNATURE_TENSOR_KEY_H

#include <unordered_set>
#include "compiler/task_signature_tensor_key.dtg.h"

namespace FlexFlow {

std::unordered_set<TaskSignatureTensorKey>
  all_keys_for_signature_arities(
    nonnegative_int num_inputs,
    nonnegative_int num_weights,
    nonnegative_int num_outputs);

} // namespace FlexFlow

#endif
