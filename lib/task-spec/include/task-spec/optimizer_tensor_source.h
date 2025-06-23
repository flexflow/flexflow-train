#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPTIMIZER_TENSOR_SOURCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPTIMIZER_TENSOR_SOURCE_H

#include "task-spec/optimizer_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct OptimizerTensorSource {
public:
  OptimizerTensorSource();

  optimizer_tensor_guid_t new_optimizer_tensor();

  void reset();

private:
  static size_t next_available_optimizer_tensor_id;
};

} // namespace FlexFlow

#endif
