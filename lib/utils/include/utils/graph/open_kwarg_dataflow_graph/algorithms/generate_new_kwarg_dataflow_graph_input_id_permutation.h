#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GENERATE_NEW_KWARG_DATAFLOW_GRAPH_INPUT_ID_PERMUTATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GENERATE_NEW_KWARG_DATAFLOW_GRAPH_INPUT_ID_PERMUTATION_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/containers/contains.h"
#include "utils/bidict/generate_bidict.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
bidict<KwargDataflowGraphInput<GraphInputName>, KwargDataflowGraphInput<GraphInputName>>
  generate_new_kwarg_dataflow_graph_input_id_permutation(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
    std::function<GraphInputName()> const &input_id_source) 
{
  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> old_graph_inputs 
    = get_all_kwarg_dataflow_graph_inputs(g);

  auto fresh_input_id = [&]() -> GraphInputName {
    while (true) {
      GraphInputName candidate = input_id_source();

      if (!contains(old_graph_inputs, KwargDataflowGraphInput{candidate})) {
        return candidate;
      }
    }
  };

  return generate_bidict(old_graph_inputs,
                         [&](KwargDataflowGraphInput<GraphInputName> const &) {
                           return KwargDataflowGraphInput<GraphInputName>{
                             fresh_input_id(),
                           };
                         })
      .reversed();
}

} // namespace FlexFlow

#endif
