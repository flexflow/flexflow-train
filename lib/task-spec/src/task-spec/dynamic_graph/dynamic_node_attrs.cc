#include "task-spec/dynamic_graph/dynamic_node_attrs.h"

namespace FlexFlow {

DynamicNodeAttrs
    decide_dynamic_node_attrs_mapping(DynamicNodeAttrs const &attrs,
                                      DynamicNodeMapping const &mapping) {
  ASSERT(!attrs.mapping.has_value());

  DynamicNodeAttrs result = attrs;
  result.mapping = mapping;

  return result;
}

} // namespace FlexFlow
