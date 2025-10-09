#include "task-spec/fwb_op_task_type.h"

namespace FlexFlow {

std::optional<OpTaskType>
  op_task_type_from_fwb_op_task_type(FwbOpTaskType fwb) {
  
  switch (fwb) {
    case FwbOpTaskType::FWD:
      return OpTaskType::FWD;
    case FwbOpTaskType::BWD:
      return OpTaskType::BWD;
    default:
      return std::nullopt;
  };
}


} // namespace FlexFlow
