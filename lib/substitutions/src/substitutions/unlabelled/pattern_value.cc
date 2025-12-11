#include "substitutions/unlabelled/pattern_value.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenKwargDataflowValue<int, TensorSlotName>
    raw_open_dataflow_value_from_pattern_value(PatternValue const &v) {
  return v.visit<
    OpenKwargDataflowValue<int, TensorSlotName>
  >(overload{
      [](PatternNodeOutput const &o) {
        return OpenKwargDataflowValue<int, TensorSlotName>{o.raw_dataflow_output};
      },
      [](PatternInput const &i) {
        return OpenDataflowValue{i.raw_dataflow_graph_input};
      },
  });
}

PatternValue
    pattern_value_from_raw_open_dataflow_value(OpenDataflowValue const &v) {
  return v.visit<PatternValue>(overload{
      [](DataflowOutput const &o) {
        return PatternValue{PatternNodeOutput{o}};
      },
      [](DataflowGraphInput const &i) { return PatternValue{PatternInput{i}}; },
  });
}

} // namespace FlexFlow
