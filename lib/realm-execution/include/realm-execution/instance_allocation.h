#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INSTANCE_ALLOCATION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INSTANCE_ALLOCATION_H

#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

bool no_instances_are_allocated(DynamicOpenDataflowGraph const &);
bool all_instances_are_allocated(DynamicOpenDataflowGraph const &);

bool instances_are_ready_for_allocation(DynamicOpenDataflowGraph const &g);

DynamicValueAttrs
    perform_instance_allocation_for_value(DynamicValueAttrs const &,
                                          Allocator &);

std::pair<DynamicOpenDataflowGraph, Realm::Event> perform_instance_allocation(
    DynamicOpenDataflowGraph const &,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    RealmContext &);

} // namespace FlexFlow

#endif
