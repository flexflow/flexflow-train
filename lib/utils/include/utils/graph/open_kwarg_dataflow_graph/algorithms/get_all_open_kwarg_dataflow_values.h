#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_OPEN_KWARG_DATAFLOW_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_OPEN_KWARG_DATAFLOW_VALUES_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<OpenKwargDataflowValue<GraphInputName, SlotName>> 
  get_all_open_kwarg_dataflow_values(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g) 
{
  std::unordered_set<KwargDataflowOutput<SlotName>>
    internal_values = get_all_kwarg_dataflow_outputs(g);

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
    external_values = get_all_kwarg_dataflow_graph_inputs(g);

  return set_union(
    transform(internal_values, 
              [](KwargDataflowOutput<SlotName> const &o) {
                return OpenKwargDataflowValue<GraphInputName, SlotName>(o); 
              }),
    transform(external_values, 
              [](KwargDataflowGraphInput<GraphInputName> const &i) {
                return OpenKwargDataflowValue<GraphInputName, SlotName>(i); 
              }));
}

} // namespace FlexFlow

#endif
