#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_GRAPH_EXPR_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_GRAPH_EXPR_H

#include "substitutions/output_graph/output_graph_expr.dtg.h"
#include "substitutions/output_graph/output_graph_expr_input.dtg.h"
#include "substitutions/output_graph/output_graph_expr_node.dtg.h"
#include "substitutions/output_graph/output_graph_expr_node_output.dtg.h"

namespace FlexFlow {

std::unordered_set<OutputGraphExprNode> get_nodes(OutputGraphExpr const &);

std::vector<OutputGraphExprNodeOutput>
    get_node_outputs(OutputGraphExpr const &, OutputGraphExprNode const &);

std::unordered_set<OutputGraphExprInput> get_inputs(OutputGraphExpr const &);

} // namespace FlexFlow

#endif
