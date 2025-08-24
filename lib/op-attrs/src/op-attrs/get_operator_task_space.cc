#include "op-attrs/get_operator_task_space.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"

namespace FlexFlow {

OperatorTaskSpace
  get_operator_task_space(ComputationGraphOpAttrs const &attrs,
                          std::vector<ParallelTensorDimDegrees> const &inputs_degrees) {
  return attrs.visit<
    OperatorTaskSpace
  >(overload {
    [&](ElementUnaryAttrs const &attrs) {
      ASSERT(inputs_degrees.size() == 1);

      return get_operator_task_space(attrs, get_only(inputs_degrees));
    },
    [&](LinearAttrs const &attrs) {
      ASSERT(inputs_degrees.size() == 1);

      return get_operator_task_space(attrs, get_only(inputs_degrees));
    },
    [](auto const &attrs) -> OperatorTaskSpace {
      PANIC("Missing implmentation of get_operator_task_space", attrs);
    },
  });
}


} // namespace FlexFlow
