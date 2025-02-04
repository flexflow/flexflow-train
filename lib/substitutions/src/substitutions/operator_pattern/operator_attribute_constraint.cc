#include "substitutions/operator_pattern/operator_attribute_constraint.h"

namespace FlexFlow {

OperatorAttributeConstraint op_type_equals_constraint(OperatorType op_type) {
  return OperatorAttributeConstraint{
      ConstraintType::EQUAL,
      OperatorAttributeExpr{OperatorAttributeKey::OP_TYPE},
      OperatorAttributeValue{op_type},
  };
}

OperatorAttributeConstraint
    op_attr_key_equals(OperatorAttributeKey key,
                       OperatorAttributeValue const &val) {
  return OperatorAttributeConstraint{
      ConstraintType::EQUAL,
      OperatorAttributeExpr{key},
      OperatorAttributeValue{val},
  };
}

OperatorAttributeConstraint
    op_attr_key_divisible_by(OperatorAttributeKey key,
                             nonnegative_int denominator) {
  return OperatorAttributeConstraint{
      ConstraintType::DIVISIBLE_BY,
      OperatorAttributeExpr{key},
      OperatorAttributeValue{denominator},
  };
}

OperatorAttributeConstraint
    make_equals_constraint(OperatorAttributeExpr const &expr,
                           OperatorAttributeValue const &val) {
  return OperatorAttributeConstraint{
      ConstraintType::EQUAL,
      expr,
      val,
  };
}

} // namespace FlexFlow
