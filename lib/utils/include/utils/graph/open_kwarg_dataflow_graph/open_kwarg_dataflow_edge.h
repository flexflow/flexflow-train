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

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowValue<GraphInputName, SlotName>
  get_src_of_open_kwarg_dataflow_edge(OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) 
{
  return e.template visit<
    OpenKwargDataflowValue<GraphInputName, SlotName>
  >(overload {
    [](KwargDataflowInputEdge<GraphInputName, SlotName> const &external_edge) {
      return OpenKwargDataflowValue<GraphInputName, SlotName>{
        external_edge.src,
      };
    },
    [](KwargDataflowEdge<SlotName> const &internal_edge) {
      return OpenKwargDataflowValue<GraphInputName, SlotName>{
        internal_edge.src,
      };
    }
  });
}

template <typename GraphInputName, typename SlotName>
KwargDataflowInput<SlotName>
  get_dst_of_open_kwarg_dataflow_edge(OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) 
{
  return e.template visit<
    KwargDataflowInput<SlotName>
  >(overload {
    [](KwargDataflowInputEdge<GraphInputName, SlotName> const &external_edge) {
      return external_edge.dst;
    },
    [](KwargDataflowEdge<SlotName> const &internal_edge) {
      return internal_edge.dst; 
    }
  });
}

} // namespace FlexFlow

#endif
