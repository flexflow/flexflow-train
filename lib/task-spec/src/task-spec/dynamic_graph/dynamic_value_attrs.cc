#include "task-spec/dynamic_graph/dynamic_value_attrs.h"

namespace FlexFlow {

DynamicValueAttrs
    decide_dynamic_value_attrs_role(DynamicValueAttrs const &attrs,
                                    DynamicTensorRole role) {
  ASSERT(attrs.role == std::nullopt);

  DynamicValueAttrs result = attrs;
  result.role = role;

  return result;
}

} // namespace FlexFlow
