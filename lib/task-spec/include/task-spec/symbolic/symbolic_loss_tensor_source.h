#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_LOSS_TENSOR_SOURCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_LOSS_TENSOR_SOURCE_H

#include "task-spec/symbolic/symbolic_loss_tensor_guid_t.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

struct SymbolicLossTensorSource {
public:
  SymbolicLossTensorSource();

  symbolic_loss_tensor_guid_t new_symbolic_loss_tensor();

private:
  static nonnegative_int next_available_symbolic_loss_tensor_id;
};

} // namespace FlexFlow

#endif
