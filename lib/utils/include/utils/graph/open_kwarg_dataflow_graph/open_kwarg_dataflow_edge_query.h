#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_EDGE_QUERY_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_input_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>
  open_kwarg_dataflow_edge_query_all() {

  return OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>{
    /*input_edge_query=*/kwarg_dataflow_input_edge_query_all<GraphInputName, SlotName>(),
    /*standard_edge_query=*/kwarg_dataflow_edge_query_all<SlotName>(),
  };
}

template <typename GraphInputName, typename SlotName>
bool open_kwarg_dataflow_edge_query_includes(
  OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &query,
  OpenKwargDataflowEdge<GraphInputName, SlotName> const &edge) 
{
  return edge.template visit<bool>(overload {
    [&](KwargDataflowInputEdge<GraphInputName, SlotName> const &input_edge) {
      return kwarg_dataflow_input_edge_query_includes(query.input_edge_query, input_edge);
    },
    [&](KwargDataflowEdge<SlotName> const &internal_edge) {
      return kwarg_dataflow_edge_query_includes(query.standard_edge_query, internal_edge); 
    }
  });
}


} // namespace FlexFlow

#endif
