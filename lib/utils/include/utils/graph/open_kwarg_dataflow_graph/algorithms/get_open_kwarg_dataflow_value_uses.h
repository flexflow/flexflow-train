#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_VALUE_USES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_VALUE_USES_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_input_edge_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<KwargDataflowInput<SlotName>>
    get_open_kwarg_dataflow_value_uses(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
        OpenKwargDataflowValue<GraphInputName, SlotName> const &v) {

  OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> query = v.template visit<
      OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>>(overload{
      [&](KwargDataflowOutput<SlotName> const &o) {
        return OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>{
            kwarg_dataflow_input_edge_query_none<GraphInputName, SlotName>(),
            KwargDataflowEdgeQuery<SlotName>{
                /*src_nodes=*/query_set<Node>{o.node},
                /*src_slots=*/query_set<SlotName>{o.slot_name},
                /*dst_nodes=*/query_set<Node>::matchall(),
                /*dst_slots=*/query_set<SlotName>::matchall(),
            },
        };
      },
      [&](KwargDataflowGraphInput<GraphInputName> const &i) {
        return OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>{
            KwargDataflowInputEdgeQuery<GraphInputName, SlotName>{
                /*srcs=*/query_set<GraphInputName>{i.name},
                /*dst_nodes=*/query_set<Node>::matchall(),
                /*dst_slots=*/query_set<SlotName>::matchall(),
            },
            kwarg_dataflow_edge_query_none<SlotName>(),
        };
      }});

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>> edges =
      g.query_edges(query);

  return transform(
      edges, [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
        return get_dst_of_open_kwarg_dataflow_edge(e);
      });
}

} // namespace FlexFlow

#endif
