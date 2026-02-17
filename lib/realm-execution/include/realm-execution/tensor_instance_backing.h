#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TENSOR_INSTANCE_BACKING_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TENSOR_INSTANCE_BACKING_H

#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"

namespace FlexFlow {

TensorInstanceBacking make_empty_tensor_instance_backing();

TensorInstanceBacking subset_tensor_instance_backing_for_invocation(
    TensorInstanceBacking const &, DynamicNodeInvocation const &);

} // namespace FlexFlow

#endif
