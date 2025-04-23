#include "substitutions/operator_pattern/satisfies_constraint.h"
#include "substitutions/operator_pattern/operator_attribute_expr.h"

namespace FlexFlow {

bool operator_satisfies_constraint(
    PCGOperatorAttrs const &attrs,
    OperatorAttributeConstraint const &constraint) {
  std::optional<OperatorAttributeValue> expr_val =
      evaluate_attribute_expr(constraint.attribute_expr, attrs);

  if (!expr_val.has_value()) {
    return false;
  }

  // std::cout << constraint.constraint_type << std::endl;
  switch (constraint.constraint_type) {
    case ConstraintType::EQUAL:
      return expr_val.value() == constraint.attribute_value;
    case ConstraintType::DIVISIBLE_BY: {
      if (expr_val.value().has<nonnegative_int>() &&
          constraint.attribute_value.has<nonnegative_int>()) {
        return expr_val.value().get<nonnegative_int>() %
                   constraint.attribute_value.get<nonnegative_int>() ==
               0;
      }
      throw mk_runtime_error(
          "DIVISIBLE_BY constraint requires nonnegative_int values");
    }
    default:
      throw mk_runtime_error(
          fmt::format("Unknown constraint type {}",
                      static_cast<int>(constraint.constraint_type)));
  }
}

} // namespace FlexFlow
