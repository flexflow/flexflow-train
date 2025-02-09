#include "substitutions/output_graph/output_graph_expr.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_graph_inputs.h"

namespace FlexFlow {

std::unordered_set<OutputGraphExprNode> get_nodes(OutputGraphExpr const &g) {
  std::unordered_set<Node> raw_nodes = get_nodes(g.raw_graph);

  return transform(raw_nodes,
                   [](Node const &n) { return OutputGraphExprNode{n}; });
}

std::vector<OutputGraphExprNodeOutput>
    get_node_outputs(OutputGraphExpr const &g, OutputGraphExprNode const &n) {
  std::vector<DataflowOutput> raw_outputs =
      get_outputs(g.raw_graph, n.raw_graph_node);

  return transform(raw_outputs, [](DataflowOutput const &o) {
    return OutputGraphExprNodeOutput{o};
  });
}

std::unordered_set<OutputGraphExprInput> get_inputs(OutputGraphExpr const &g) {
  std::unordered_set<DataflowGraphInput> raw_inputs =
      get_open_dataflow_graph_inputs(g.raw_graph);

  return transform(raw_inputs, [](DataflowGraphInput const &i) {
    return OutputGraphExprInput{i};
  });
}

} // namespace FlexFlow
