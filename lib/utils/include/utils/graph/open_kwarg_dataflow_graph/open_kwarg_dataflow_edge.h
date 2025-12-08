#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_EDGE_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowEdge<GraphInputName, SlotName>
  mk_open_kwarg_dataflow_edge_from_src_val_and_dst(
    OpenKwargDataflowValue<GraphInputName, SlotName> const &src,
    KwargDataflowInput<SlotName> const &dst)
{
  return src.template visit<
    OpenKwargDataflowEdge<GraphInputName, SlotName>
  >(overload {
    [&](KwargDataflowOutput<SlotName> const &output) {
      return OpenKwargDataflowEdge<GraphInputName, SlotName>{
        KwargDataflowEdge<SlotName>{
          /*src=*/output,
          /*dst=*/dst,
        },
      };
    },
    [&](KwargDataflowGraphInput<GraphInputName> const &graph_input) {
      return OpenKwargDataflowEdge<GraphInputName, SlotName>{
        KwargDataflowInputEdge<GraphInputName, SlotName>{
          /*src=*/graph_input,
          /*dst=*/dst,
        },
      };
    }
  });
}

} // namespace FlexFlow

#endif
