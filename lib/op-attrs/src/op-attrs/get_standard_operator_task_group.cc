#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/split.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/upsample.h"
#include "op-attrs/ops/weight.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_two_keys.h"
#include "utils/overload.h"

namespace FlexFlow {

StandardOperatorTaskGroup get_standard_operator_task_group(
    ComputationGraphOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const
        &input_dim_degree_binding) {
  return op_attrs.visit<StandardOperatorTaskGroup>(overload{
      [&](BatchNormAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return batch_norm_get_task_group(attrs, input_dim_degrees);
      },
      [&](BatchMatmulAttrs const &attrs) -> StandardOperatorTaskGroup {
        auto [lhs_input_dim_degrees, rhs_input_dim_degrees] =
            require_two_keys(input_dim_degree_binding,
                             TensorSlotName::LHS_INPUT,
                             TensorSlotName::RHS_INPUT);

        return batch_matmul_get_task_group(
            attrs, lhs_input_dim_degrees, rhs_input_dim_degrees);
      },
      [&](ElementBinaryAttrs const &attrs) -> StandardOperatorTaskGroup {
        auto [lhs_input_dim_degrees, rhs_input_dim_degrees] =
            require_two_keys(input_dim_degree_binding,
                             TensorSlotName::LHS_INPUT,
                             TensorSlotName::RHS_INPUT);

        return element_binary_get_task_group(
            attrs, lhs_input_dim_degrees, rhs_input_dim_degrees);
      },
      [&](ElementUnaryAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return element_unary_get_task_group(attrs, input_dim_degrees);
      },
      [&](FlatAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return flat_get_task_group(attrs, input_dim_degrees);
      },
      [&](InputAttrs const &attrs) -> StandardOperatorTaskGroup {
        ASSERT(input_dim_degree_binding.empty());

        return input_get_task_group(attrs);
      },
      [&](LinearAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return linear_get_task_group(attrs, input_dim_degrees);
      },
      [&](SoftmaxAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return softmax_get_task_group(attrs, input_dim_degrees);
      },
      [&](SplitAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return split_get_task_group(attrs, input_dim_degrees);
      },
      [&](TransposeAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return transpose_get_task_group(attrs, input_dim_degrees);
      },
      [&](UpsampleAttrs const &attrs) -> StandardOperatorTaskGroup {
        ParallelTensorDimDegrees input_dim_degrees =
            require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

        return upsample_get_task_group(attrs, input_dim_degrees);
      },
      [&](WeightAttrs const &attrs) -> StandardOperatorTaskGroup {
        ASSERT(input_dim_degree_binding.empty());

        return weight_get_task_group(attrs);
      },
      [&](auto const &) -> StandardOperatorTaskGroup { PANIC(); },
  });
}

} // namespace FlexFlow
