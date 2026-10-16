#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/ops/element_unary.h"
#include "utils/containers/require_only_key.h"
#include "utils/overload.h"

namespace FlexFlow {

StandardOperatorTaskGroup get_standard_operator_task_group(
    ComputationGraphOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const
        &input_dim_degree_binding) {
  return op_attrs.visit<StandardOperatorTaskGroup>(overload{
      [&](ElementUnaryAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return element_unary_get_task_group(attrs, input_dim_degrees);
      },
      [&](auto const &) -> StandardOperatorTaskGroup { PANIC(); },
  });
}

} // namespace FlexFlow
