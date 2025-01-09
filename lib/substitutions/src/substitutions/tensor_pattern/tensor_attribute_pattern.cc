#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

TensorAttributePattern tensor_attribute_pattern_match_all() {
  return TensorAttributePattern{{}};
}

TensorAttributePattern tensor_attr_pattern_require_num_dims(int num_dims) {
  return TensorAttributePattern{{
      TensorAttributeConstraint{
          ConstraintType::EQUAL,
          TensorAttributeExpr{
              TensorAttributeListSize{
                  TensorAttributeKey::DIM_SIZES,
              },
          },
          TensorAttributeValue{size_t_from_int(num_dims)},
      },
  }};
}

} // namespace FlexFlow
