#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_FORWARD_TENSOR_SOURCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_FORWARD_TENSOR_SOURCE_H

#include "task-spec/forward_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct ForwardTensorSource {
public:
  ForwardTensorSource();

  forward_tensor_guid_t new_forward_tensor();

  void reset();

private:
  static int next_available_forward_tensor_id;
};

} // namespace FlexFlow

#endif
