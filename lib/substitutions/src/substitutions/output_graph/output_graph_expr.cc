#include "substitutions/output_graph/output_graph_expr.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"

namespace FlexFlow {

std::unordered_set<OutputGraphExprNode> get_nodes(OutputGraphExpr const &g) {
  std::unordered_set<Node> raw_nodes = get_nodes(g.raw_graph);

  return transform(raw_nodes,
                   [](Node const &n) { return OutputGraphExprNode{n}; });
}

std::unordered_map<TensorSlotName, OutputGraphExprNodeOutput>
    get_node_outputs(OutputGraphExpr const &g, OutputGraphExprNode const &n) {
  std::unordered_map<TensorSlotName, KwargDataflowOutput<TensorSlotName>>
      raw_outputs = get_outgoing_kwarg_dataflow_outputs_for_node(
          g.raw_graph, n.raw_graph_node);

  return map_values(raw_outputs,
                    [](KwargDataflowOutput<TensorSlotName> const &o) {
                      return OutputGraphExprNodeOutput{o};
                    });
}

std::unordered_set<OutputGraphExprInput> get_inputs(OutputGraphExpr const &g) {
  std::unordered_set<KwargDataflowGraphInput<int>> raw_inputs =
      get_all_kwarg_dataflow_graph_inputs(g.raw_graph);

  return transform(raw_inputs, [](KwargDataflowGraphInput<int> const &i) {
    return OutputGraphExprInput{i};
  });
}

} // namespace FlexFlow
