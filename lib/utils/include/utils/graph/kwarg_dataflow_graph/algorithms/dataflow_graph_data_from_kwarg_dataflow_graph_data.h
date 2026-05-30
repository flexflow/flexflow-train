#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_DATA_FROM_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_DATA_FROM_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/containers/group_by.h"
#include "utils/containers/index_of.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_data.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.dtg.h"
#include "utils/one_to_many/one_to_many_transform_values.h"

namespace FlexFlow {

template <typename SlotName>
DataflowGraphData dataflow_graph_data_from_kwarg_dataflow_graph_data(
    KwargDataflowGraphData<SlotName> const &kwarg_data,
    std::function<std::vector<SlotName>(
        std::unordered_set<SlotName> const &)> const &order_slots) {
  std::unordered_set<KwargDataflowInput<SlotName>> all_inputs = transform(
      kwarg_data.edges,
      [](KwargDataflowEdge<SlotName> const &e) -> KwargDataflowInput<SlotName> {
        return e.dst;
      });

  std::unordered_set<KwargDataflowOutput<SlotName>> all_outputs =
      kwarg_data.outputs;

  std::unordered_map<Node, std::unordered_set<SlotName>>
      incoming_slots_by_node = map_values(
          group_by(all_inputs,
                   [](KwargDataflowInput<SlotName> const &i) -> Node {
                     return i.node;
                   })
              .l_to_r(),
          [](nonempty_unordered_set<KwargDataflowInput<SlotName>> const &is)
              -> std::unordered_set<SlotName> {
            return transform(is.unwrap_as_unordered_set(),
                             [](KwargDataflowInput<SlotName> const &i) {
                               return i.slot_name;
                             });
          });

  std::unordered_map<Node, std::unordered_set<SlotName>>
      outgoing_slots_by_node = map_values(
          group_by(all_outputs,
                   [](KwargDataflowOutput<SlotName> const &o) -> Node {
                     return o.node;
                   })
              .l_to_r(),
          [](nonempty_unordered_set<KwargDataflowOutput<SlotName>> const &os)
              -> std::unordered_set<SlotName> {
            return transform(os.unwrap_as_unordered_set(),
                             [](KwargDataflowOutput<SlotName> const &o) {
                               return o.slot_name;
                             });
          });

  auto dataflow_input_from_kwarg_input =
      [&](KwargDataflowInput<SlotName> const &i) -> DataflowInput {
    std::vector<SlotName> slot_ordering =
        order_slots(incoming_slots_by_node.at(i.node));

    return DataflowInput{
        /*node=*/i.node,
        /*idx=*/
        nonnegative_int{
            index_of(slot_ordering, i.slot_name).value(),
        },
    };
  };

  auto dataflow_output_from_kwarg_output =
      [&](KwargDataflowOutput<SlotName> const &o) -> DataflowOutput {
    std::vector<SlotName> slot_ordering =
        order_slots(outgoing_slots_by_node.at(o.node));

    return DataflowOutput{
        /*node=*/o.node,
        /*idx=*/
        nonnegative_int{
            index_of(slot_ordering, o.slot_name).value(),
        },
    };
  };

  auto dataflow_edge_from_kwarg_dataflow_edge =
      [&](KwargDataflowEdge<SlotName> const &kwarg_edge) -> DataflowEdge {
    return DataflowEdge{
        /*src=*/dataflow_output_from_kwarg_output(kwarg_edge.src),
        /*dst=*/dataflow_input_from_kwarg_input(kwarg_edge.dst),
    };
  };

  DataflowGraphData result_data = DataflowGraphData{
      /*nodes=*/kwarg_data.nodes,
      /*edges=*/
      transform(kwarg_data.edges, dataflow_edge_from_kwarg_dataflow_edge),
      /*outputs=*/
      transform(kwarg_data.outputs, dataflow_output_from_kwarg_output),
  };

  return result_data;
}

} // namespace FlexFlow

#endif
