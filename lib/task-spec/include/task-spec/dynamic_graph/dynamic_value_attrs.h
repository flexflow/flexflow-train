#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_VALUE_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_VALUE_ATTRS_H

#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"

namespace FlexFlow {

DynamicValueAttrs decide_dynamic_value_attrs_role(DynamicValueAttrs const &,
                                                  DynamicTensorRole);

} // namespace FlexFlow

#endif
