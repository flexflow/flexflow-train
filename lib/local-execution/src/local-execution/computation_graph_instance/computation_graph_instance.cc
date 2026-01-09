#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph_from_cg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "utils/exception.h"

namespace FlexFlow {

ComputationGraphInstance::ComputationGraphInstance(
    DynamicOpenDataflowGraph const &graph, LocalTaskRegistry const &registry)
    : expanded_dataflow_graph(graph), task_registry(registry) {}

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &cg,
    bidict<tensor_guid_t, // FIXME (Elliott): not sure this is correct
           std::variant<GenericTensorAccessorW, GenericTensorAccessorR>> const
        &tensors,
    LocalTaskRegistry registry) {
  DynamicOpenDataflowGraph dg = make_dynamic_open_dataflow_graph_from_cg(cg);
  dg = perform_pass_expansion(dg);
  dg = perform_shard_expansion(dg);
  // dg = perform_update_insertion(dg);
  return ComputationGraphInstance{dg, registry};
}

} // namespace FlexFlow
