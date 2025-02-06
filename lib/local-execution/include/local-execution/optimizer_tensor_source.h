#ifndef _FLEXFLOW_LOCAL_EXECUTION_OPTIMIZER_TENSOR_SOURCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_OPTIMIZER_TENSOR_SOURCE_H

#include "task-spec/optimizer_tensor_t.dtg.h"

namespace FlexFlow {

struct OptimizerTensorSource {
public:
  OptimizerTensorSource();

  optimizer_tensor_t new_optimizer_tensor();

private:
  static size_t next_available_optimizer_tensor_id;
};

} // namespace FlexFlow

#endif
