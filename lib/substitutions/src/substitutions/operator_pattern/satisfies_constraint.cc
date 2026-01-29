#include "substitutions/operator_pattern/satisfies_constraint.h"
#include "substitutions/operator_pattern/operator_attribute_expr.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

bool operator_satisfies_constraint(
    PCGOperatorAttrs const &attrs,
    OperatorAttributeConstraint const &constraint) {
  std::optional<OperatorAttributeValue> expr_val =
      evaluate_attribute_expr(constraint.attribute_expr, attrs);

  if (!expr_val.has_value()) {
    return false;
  }

  switch (constraint.constraint_type) {
    case ConstraintType::EQUAL:
      return expr_val.value() == constraint.attribute_value;
    case ConstraintType::DIVISIBLE_BY: {
      ASSERT(expr_val.value().has<nonnegative_int>() &&
              constraint.attribute_value.has<nonnegative_int>(),
            "DIVISIBLE_BY constraint requires nonnegative_int values");

      return expr_val.value().get<nonnegative_int>() %
                 constraint.attribute_value.get<nonnegative_int>() ==
             0;
    }
    default:
      PANIC("Unknown constraint type", constraint.constraint_type);
  }
}

} // namespace FlexFlow
