#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_NODE_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_NODE_IDS_H

#include "utils/containers/generate_map.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_view_with_labelling.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/permute_open_kwarg_dataflow_graph_node_ids.h"
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
    permute_labelled_open_kwarg_dataflow_graph_node_ids(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &g,
        bidict<NewNode, Node> const &new_node_tofrom_old_node) {

  OpenKwargDataflowGraphView<GraphInputName, SlotName> permuted =
      permute_open_kwarg_dataflow_graph_node_ids(
          static_cast<OpenKwargDataflowGraphView<GraphInputName, SlotName>>(g),
          new_node_tofrom_old_node);

  auto old_node_from_new = [&](Node const &new_node) {
    return new_node_tofrom_old_node.at_l(NewNode{new_node});
  };

  auto old_value_from_new =
      [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &new_value) {
        return new_value
            .template visit<OpenKwargDataflowValue<GraphInputName, SlotName>>(
                overload{
                    [&](KwargDataflowOutput<SlotName> const &new_o) {
                      return OpenKwargDataflowValue<GraphInputName, SlotName>{
                          KwargDataflowOutput<SlotName>{
                              old_node_from_new(new_o.node),
                              new_o.slot_name,
                          },
                      };
                    },
                    [](KwargDataflowGraphInput<GraphInputName> const &i) {
                      return OpenKwargDataflowValue<GraphInputName, SlotName>{
                          i};
                    },
                });
      };

  std::unordered_map<Node, NodeLabel> node_labels =
      generate_map(get_nodes(permuted), [&](Node const &new_node) {
        return g.at(old_node_from_new(new_node));
      });

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
