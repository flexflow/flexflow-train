#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_GRADIENT_TENSOR_SOURCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_GRADIENT_TENSOR_SOURCE_H

#include "task-spec/gradient_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct GradientTensorSource {
public:
  GradientTensorSource();

  gradient_tensor_guid_t new_gradient_tensor();

  void reset();

private:
  static size_t next_available_gradient_tensor_id;
};

} // namespace FlexFlow

#endif
