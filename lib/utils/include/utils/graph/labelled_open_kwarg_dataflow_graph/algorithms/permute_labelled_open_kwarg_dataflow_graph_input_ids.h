#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_INPUT_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_INPUT_IDS_H

#include "utils/containers/generate_map.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_view_with_labelling.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/permute_open_kwarg_dataflow_graph_input_ids.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
LabelledOpenKwargDataflowGraphView<NodeLabel,
                                   ValueLabel,
                                   GraphInputName,
                                   SlotName>
    permute_labelled_open_kwarg_dataflow_graph_input_ids(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &g,
        bidict<KwargDataflowGraphInput<GraphInputName>,
               KwargDataflowGraphInput<GraphInputName>> const
            &new_input_to_old_input) {

  OpenKwargDataflowGraphView<GraphInputName, SlotName> permuted =
      permute_open_kwarg_dataflow_graph_input_ids(
          static_cast<OpenKwargDataflowGraphView<GraphInputName, SlotName>>(g),
          new_input_to_old_input);

  auto old_input_from_new =
      [&](KwargDataflowGraphInput<GraphInputName> const &i)
      -> KwargDataflowGraphInput<GraphInputName> {
    return new_input_to_old_input.at_l(i);
  };

  auto old_value_from_new =
      [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &new_value) {
        return new_value
            .template visit<OpenKwargDataflowValue<GraphInputName, SlotName>>(
                overload{
                    [](KwargDataflowOutput<SlotName> const &o) {
                      return OpenKwargDataflowValue<GraphInputName, SlotName>{
                          o};
                    },
                    [&](KwargDataflowGraphInput<GraphInputName> const &i) {
                      return OpenKwargDataflowValue<GraphInputName, SlotName>{
                          old_input_from_new(i)};
                    },
                });
      };

  std::unordered_map<Node, NodeLabel> node_labels =
      generate_map(get_nodes(permuted), [&](Node const &n) { return g.at(n); });

  std::unordered_map<OpenKwargDataflowValue<GraphInputName, SlotName>,
                     ValueLabel>
      value_labels = generate_map(
          get_all_open_kwarg_dataflow_values(permuted),
          [&](OpenKwargDataflowValue<GraphInputName, SlotName> const
                  &new_value) { return g.at(old_value_from_new(new_value)); });

  return open_kwarg_dataflow_graph_view_with_labelling(
      permuted, node_labels, value_labels);
}

} // namespace FlexFlow

#endif
