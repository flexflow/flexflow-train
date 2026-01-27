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
    case ConstraintType::DIVISIBLE_BY: {
      if (expr_val.has<nonnegative_int>() &&
          constraint.attribute_value.has<nonnegative_int>()) {
        return expr_val.get<nonnegative_int>() %
                   constraint.attribute_value.get<nonnegative_int>() ==
               0;
      }
      throw mk_runtime_error(
          "DIVISIBLE_BY constraint requires nonnegative_int values");
    }
    default:
      PANIC("Unknown constraint type", constraint.constraint_type);
  }
}

} // namespace FlexFlow
