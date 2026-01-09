#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/exception.h"

namespace FlexFlow {

ComputationGraphInstance::ComputationGraphInstance(
    DynamicOpenDataflowGraph const &graph, LocalTaskRegistry const &registry)
    : expanded_dataflow_graph(graph), task_registry(registry) {
  NOT_IMPLEMENTED();
}

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &cg,
    bidict<tensor_guid_t, // FIXME (Elliott): not sure this is correct
           std::variant<GenericTensorAccessorW, GenericTensorAccessorR>> const
        &tensors,
    LocalTaskRegistry registry) {
  DynamicOpenDataflowGraph dg{make_empty_dynamic_open_dataflow_graph()};
  return ComputationGraphInstance{dg, registry};
}

} // namespace FlexFlow
