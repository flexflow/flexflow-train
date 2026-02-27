#include "task-spec/dynamic_graph/dynamic_task_type.h"
#include "utils/overload.h"

namespace FlexFlow {

DynamicTaskType decide_copy_task_type(DynamicTensorRole role) {
  return role.visit<DynamicTaskType>(overload{
      [](FwbTensorType const &fwb_tensor) {
        switch (fwb_tensor) {
          case FwbTensorType::FORWARD:
            return DynamicTaskType::FWD;
          case FwbTensorType::GRADIENT:
            return DynamicTaskType::BWD;
          default:
            PANIC("Unexpected FwbTensorType", fwb_tensor);
            break;
        }
      },
      [](DynamicOptimizerTensorRole const &) { return DynamicTaskType::UPD; },
      [](DynamicLossTensorRole const &) { return DynamicTaskType::LOSS; },
  });
}

} // namespace FlexFlow
