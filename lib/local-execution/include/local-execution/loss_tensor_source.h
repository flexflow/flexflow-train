#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOSS_TENSOR_SOURCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOSS_TENSOR_SOURCE_H

#include "local-execution/loss_tensor_t.dtg.h"

namespace FlexFlow {

struct LossTensorSource {
public:
  LossTensorSource();

  loss_tensor_t new_loss_tensor();

private:
  static size_t next_available_loss_tensor_id;
};

} // namespace FlexFlow

#endif
