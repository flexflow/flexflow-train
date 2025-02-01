#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOWERED_TENSOR_SOURCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOWERED_TENSOR_SOURCE_H

#include "local-execution/lowered_tensor_t.dtg.h"

namespace FlexFlow {

struct LoweredTensorSource {
public:
  LoweredTensorSource();

  lowered_tensor_t new_lowered_tensor();

private:
  static size_t next_available_lowered_tensor_id;
};

} // namespace FlexFlow

#endif
