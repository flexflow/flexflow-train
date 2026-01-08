#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_ISOMORPHISM_H

#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowValue<GraphInputName, SlotName>
    isomorphism_map_r_open_kwarg_dataflow_value_from_l(
        OpenKwargDataflowGraphIsomorphism<GraphInputName> const &iso,
        OpenKwargDataflowValue<GraphInputName, SlotName> const &l_value) {
  return l_value
      .template visit<OpenKwargDataflowValue<GraphInputName, SlotName>>(
          overload{
              [&](KwargDataflowGraphInput<GraphInputName> const &l_input) {
                return OpenKwargDataflowValue<GraphInputName, SlotName>{
                    iso.input_mapping.at_l(l_input),
                };
              },
              [&](KwargDataflowOutput<SlotName> const &l_output) {
                return OpenKwargDataflowValue<GraphInputName, SlotName>{
                    isomorphism_map_r_kwarg_dataflow_output_from_l(iso,
                                                                   l_output),
                };
              },
          });
}

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowValue<GraphInputName, SlotName>
    isomorphism_map_l_open_kwarg_dataflow_value_from_r(
        OpenKwargDataflowGraphIsomorphism<GraphInputName> const &iso,
        OpenKwargDataflowValue<GraphInputName, SlotName> const &r_value) {
  return r_value
      .template visit<OpenKwargDataflowValue<GraphInputName, SlotName>>(
          overload{
              [&](KwargDataflowGraphInput<GraphInputName> const &r_input) {
                return OpenKwargDataflowValue<GraphInputName, SlotName>{
                    iso.input_mapping.at_r(r_input),
                };
              },
              [&](KwargDataflowOutput<SlotName> const &r_output) {
                return OpenKwargDataflowValue<GraphInputName, SlotName>{
                    isomorphism_map_l_kwarg_dataflow_output_from_r(iso,
                                                                   r_output),
                };
              },
          });
}

template <typename GraphInputName, typename SlotName>
KwargDataflowOutput<SlotName> isomorphism_map_r_kwarg_dataflow_output_from_l(
    OpenKwargDataflowGraphIsomorphism<GraphInputName> const &iso,
    KwargDataflowOutput<SlotName> const &l_output) {
  return KwargDataflowOutput<SlotName>{
      iso.node_mapping.at_l(l_output.node),
      l_output.slot_name,
  };
}

template <typename GraphInputName, typename SlotName>
KwargDataflowOutput<SlotName> isomorphism_map_l_kwarg_dataflow_output_from_r(
    OpenKwargDataflowGraphIsomorphism<GraphInputName> const &iso,
    KwargDataflowOutput<SlotName> const &r_output) {
  return KwargDataflowOutput<SlotName>{
      iso.node_mapping.at_r(r_output.node),
      r_output.slot_name,
  };
}

} // namespace FlexFlow

#endif
