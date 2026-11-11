#include "task-spec/dynamic_graph/serializable_dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"

namespace FlexFlow {

SerializableDynamicOpenDataflowGraph
    dynamic_open_dataflow_graph_to_serializable(
        DynamicOpenDataflowGraph const &g) {
  return SerializableDynamicOpenDataflowGraph{
      /*invocations=*/transform(get_dynamic_invocation_set(g),
                                dynamic_node_invocation_to_serializable),
  };
}

DynamicOpenDataflowGraph dynamic_open_dataflow_graph_from_serializable(
    SerializableDynamicOpenDataflowGraph const &serializable) {
  return DynamicOpenDataflowGraph{
      /*invocations=*/transform(serializable.invocations,
                                dynamic_node_invocation_from_serializable),
  };
}

} // namespace FlexFlow
