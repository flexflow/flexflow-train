#include "substitutions/output_graph/output_graph_expr_value.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenDataflowValue raw_open_dataflow_value_from_output_graph_expr_value(
    OutputGraphExprValue const &v) {
  return v.visit<OpenDataflowValue>(overload{
      [](OutputGraphExprNodeOutput const &o) {
        return OpenDataflowValue{o.raw_dataflow_output};
      },
      [](OutputGraphExprInput const &i) {
        return OpenDataflowValue{i.raw_dataflow_graph_input};
      },
  });
}

OutputGraphExprValue output_graph_expr_value_from_raw_open_dataflow_value(
    OpenDataflowValue const &v) {
  return v.visit<OutputGraphExprValue>(overload{
      [](DataflowOutput const &o) {
        return OutputGraphExprValue{OutputGraphExprNodeOutput{o}};
      },
      [](DataflowGraphInput const &i) {
        return OutputGraphExprValue{OutputGraphExprInput{i}};
      },
  });
}

} // namespace FlexFlow
