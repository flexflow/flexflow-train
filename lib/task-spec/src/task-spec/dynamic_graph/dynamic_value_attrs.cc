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

DynamicValueAttrs
    dynamic_value_attrs_with_mapping(DynamicValueAttrs const &v,
                                     ParallelTensorMapping const &m) {
  ASSERT(v.mapping == std::nullopt);
  DynamicValueAttrs result = v;
  result.mapping = m;
  return result;
}

} // namespace FlexFlow
