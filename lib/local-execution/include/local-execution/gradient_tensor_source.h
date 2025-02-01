#ifndef _FLEXFLOW_LOCAL_EXECUTION_GRADIENT_TENSOR_SOURCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_GRADIENT_TENSOR_SOURCE_H

#include "local-execution/gradient_tensor_t.dtg.h"

namespace FlexFlow {

struct GradientTensorSource {
public:
  GradientTensorSource();

  gradient_tensor_t new_gradient_tensor();

private:
  static size_t next_available_gradient_tensor_id;
};

} // namespace FlexFlow

#endif
