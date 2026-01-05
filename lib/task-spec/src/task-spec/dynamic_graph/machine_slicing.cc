#include "task-spec/dynamic_graph/machine_slicing.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"

namespace FlexFlow {

std::unordered_set<DynamicNodeInvocation> 
  perform_machine_slicing_for_invocation(DynamicNodeInvocation const &invocation,
                                         MachineSpaceCoordinate const &device_coord) {

  ASSERT(invocation.node_attrs.device_coord.has_value());

  if (invocation.node_attrs.device_coord.value() == device_coord) {
    return {invocation};
  } else {
    return {};
  }
}

DynamicOpenDataflowGraph perform_machine_slicing(DynamicOpenDataflowGraph const &g,
                                                 MachineSpaceCoordinate const &device_coord) {
  DynamicOpenDataflowGraph result = flatmap_dynamic_invocation_set(
    g, 
    [&](DynamicNodeInvocation const &invocation) 
      -> std::unordered_set<DynamicNodeInvocation>
    {
      return perform_machine_slicing_for_invocation(invocation, device_coord);
    });

  return result;
}

} // namespace FlexFlow
