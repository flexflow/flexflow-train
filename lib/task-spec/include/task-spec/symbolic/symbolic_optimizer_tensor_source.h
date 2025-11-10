#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_OPTIMIZER_TENSOR_SOURCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_OPTIMIZER_TENSOR_SOURCE_H

#include "task-spec/symbolic/symbolic_optimizer_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct SymbolicOptimizerTensorSource {
public:
  SymbolicOptimizerTensorSource();

  symbolic_optimizer_tensor_guid_t new_symbolic_optimizer_tensor();

  void reset();

private:
  static int next_available_symbolic_optimizer_tensor_id;
};

} // namespace FlexFlow

#endif
