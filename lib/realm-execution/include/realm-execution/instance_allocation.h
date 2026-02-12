#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INSTANCE_ALLOCATION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INSTANCE_ALLOCATION_H

#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

std::pair<Realm::RegionInstance, Realm::Event>
    perform_instance_allocation_for_value(DynamicNodeAttrs const &node,
                                          DynamicValueAttrs const &value,
                                          RealmContext &ctx);

TensorInstanceBacking perform_instance_allocation(
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    RealmContext &ctx);

} // namespace FlexFlow

#endif
