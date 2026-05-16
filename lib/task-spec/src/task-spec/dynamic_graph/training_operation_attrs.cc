#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/overload.h"

namespace FlexFlow {

bool training_op_attrs_has_op_type(TrainingOperationAttrs const &op_attrs, OperatorType op_type) {
  return op_attrs.visit<bool>(overload {
    [&](PCGOperatorAttrs const &pcg_op_attrs) -> bool {
      return pcg_op_attrs_get_op_type(pcg_op_attrs) == op_type;
    },
    [](LossAttrs const &) -> bool {
      return false;
    },
    [](CopyAttrs const &) -> bool {
      return false;
    },
  });
}

} // namespace FlexFlow
