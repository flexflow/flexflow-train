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

  switch (constraint.constraint_type) {
    case ConstraintType::EQUAL:
      return expr_val.value() == constraint.attribute_value;
    case ConstraintType::DIVISIBLE_BY: {
      auto get_nonnegative_int_if_possible =
          [](OperatorAttributeValue v) -> std::optional<nonnegative_int> {
        if (v.has<nonnegative_int>()) {
          return v.get<nonnegative_int>();
        }
        if (v.has<positive_int>()) {
          return v.get<positive_int>().nonnegative_int_from_positive_int();
        }
        return std::nullopt;
      };

      if (!expr_val.has_value()) {
        throw mk_runtime_error("DIVISIBLE_BY constraint requires "
                               "nonnegative_int or positive_int values");
      }

      std::optional<nonnegative_int> maybe_expr_val_nn =
          get_nonnegative_int_if_possible(expr_val.value());
      std::optional<nonnegative_int> maybe_attr_val_nn =
          get_nonnegative_int_if_possible(constraint.attribute_value);

      if (maybe_expr_val_nn.has_value() && maybe_attr_val_nn.has_value()) {
        return maybe_expr_val_nn.value() % maybe_attr_val_nn.value() == 0;
      }
      throw mk_runtime_error("DIVISIBLE_BY constraint requires nonnegative_int "
                             "or positive_int values");
    }
    default:
      throw mk_runtime_error(
          fmt::format("Unknown constraint type {}",
                      static_cast<int>(constraint.constraint_type)));
  }
}

} // namespace FlexFlow
