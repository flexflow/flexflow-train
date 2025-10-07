#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_FORWARD_TRAINING_SOURCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_FORWARD_TRAINING_SOURCE_H

#include "task-spec/symbolic_forward_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct SymbolicForwardTensorSource {
public:
  SymbolicForwardTensorSource();

  symbolic_forward_tensor_guid_t new_symbolic_forward_tensor();

  void reset();

private:
  static int next_available_symbolic_forward_tensor_id;
};

} // namespace FlexFlow

#endif
