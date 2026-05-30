#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_VALUE_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_VALUE_ATTRS_H

#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.dtg.h"

namespace FlexFlow {

DynamicValueAttrs decide_dynamic_value_attrs_role(DynamicValueAttrs const &,
                                                  DynamicTensorRole);

DynamicValueAttrs
    dynamic_value_attrs_with_mapping(DynamicValueAttrs const &,
                                     ParallelTensorMapping const &);

} // namespace FlexFlow

#endif
