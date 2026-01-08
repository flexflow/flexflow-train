#include "substitutions/tensor_pattern/satisfies_constraint.h"
#include "substitutions/tensor_pattern/tensor_attribute_expr.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

bool parallel_tensor_satisfies_constraint(
    ParallelTensorAttrs const &attrs,
    TensorAttributeConstraint const &constraint) {
  TensorAttributeValue expr_val =
      evaluate_attribute_expr(attrs, constraint.attribute_expr);

  switch (constraint.constraint_type) {
    case ConstraintType::EQUAL:
      return expr_val == constraint.attribute_value;
    default:
      PANIC("Unknown constraint type", constraint.constraint_type);
  }
}

} // namespace FlexFlow
