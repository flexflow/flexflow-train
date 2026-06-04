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

DynamicValueAttrs decide_dynamic_value_attrs_mapping(
    DynamicValueAttrs const &attrs,
    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const &mapping)
{
  ASSERT(!attrs.mapping.has_value());

  DynamicValueAttrs result = attrs;
  result.mapping = mapping;

  return result;
}


} // namespace FlexFlow
