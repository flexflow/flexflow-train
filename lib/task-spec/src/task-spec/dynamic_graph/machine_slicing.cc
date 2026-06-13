#include "task-spec/dynamic_graph/machine_slicing.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"

namespace FlexFlow {

std::unordered_set<DynamicNodeInvocation>
    perform_machine_slicing_for_invocation(
        DynamicNodeInvocation const &invocation,
        global_device_id_t const &device_id) {

  ASSERT(invocation.node_attrs.device_id.has_value());

  if (invocation.node_attrs.device_id.value() == device_id) {
    return {invocation};
  } else {
    return {};
  }
}

DynamicOpenDataflowGraph
    perform_machine_slicing(DynamicOpenDataflowGraph const &g,
                            global_device_id_t const &device_id) {
  DynamicOpenDataflowGraph result = flatmap_dynamic_invocation_set(
      g,
      [&](DynamicNodeInvocation const &invocation)
          -> std::unordered_set<DynamicNodeInvocation> {
        return perform_machine_slicing_for_invocation(invocation, device_id);
      });

  return result;
}

} // namespace FlexFlow
