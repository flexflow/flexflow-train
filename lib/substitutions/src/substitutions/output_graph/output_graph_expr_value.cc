#include "substitutions/output_graph/output_graph_expr_value.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenKwargDataflowValue<int, TensorSlotName> raw_open_dataflow_value_from_output_graph_expr_value(
    OutputGraphExprValue const &v) {
  return v.visit<OpenKwargDataflowValue<int, TensorSlotName>>(overload{
      [](OutputGraphExprNodeOutput const &o) {
        return OpenKwargDataflowValue<int, TensorSlotName>{o.raw_dataflow_output};
      },
      [](OutputGraphExprInput const &i) {
        return OpenKwargDataflowValue<int, TensorSlotName>{i.raw_dataflow_graph_input};
      },
  });
}

OutputGraphExprValue output_graph_expr_value_from_raw_open_dataflow_value(
    OpenKwargDataflowValue<int, TensorSlotName> const &v) {
  return v.visit<OutputGraphExprValue>(overload{
      [](KwargDataflowOutput<TensorSlotName> const &o) {
        return OutputGraphExprValue{OutputGraphExprNodeOutput{o}};
      },
      [](KwargDataflowGraphInput<int> const &i) {
        return OutputGraphExprValue{OutputGraphExprInput{i}};
      },
  });
}

} // namespace FlexFlow
