#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_DATA_FROM_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_DATA_FROM_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/containers/group_by.h"
#include "utils/containers/index_of.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_data.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.dtg.h"
#include "utils/nonempty_unordered_set/nonempty_unordered_set.h"
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

  OneToMany<Node, SlotName> incoming_slots_by_node =
      one_to_many_transform_values(
          group_by(all_inputs,
                   [](KwargDataflowInput<SlotName> const &i) -> Node {
                     return i.node;
                   }),
          [](KwargDataflowInput<SlotName> const &i) -> SlotName {
            return i.slot_name;
          });

  OneToMany<Node, SlotName> outgoing_slots_by_node =
      one_to_many_transform_values(
          group_by(all_outputs,
                   [](KwargDataflowOutput<SlotName> const &o) -> Node {
                     return o.node;
                   }),
          [](KwargDataflowOutput<SlotName> const &o) -> SlotName {
            return o.slot_name;
          });

  auto dataflow_input_from_kwarg_input =
      [&](KwargDataflowInput<SlotName> const &i) -> DataflowInput {
    std::vector<SlotName> slot_ordering = order_slots(
        incoming_slots_by_node.at_l(i.node).unwrap_as_unordered_set());

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
    std::vector<SlotName> slot_ordering = order_slots(
        outgoing_slots_by_node.at_l(o.node).unwrap_as_unordered_set());

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
