#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_GRADIENT_TENSOR_SOURCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_GRADIENT_TENSOR_SOURCE_H

#include "task-spec/symbolic/symbolic_gradient_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct SymbolicGradientTensorSource {
public:
  SymbolicGradientTensorSource();

  symbolic_gradient_tensor_guid_t new_symbolic_gradient_tensor();

  void reset();

private:
  static int next_available_symbolic_gradient_tensor_id;
};

} // namespace FlexFlow

#endif
