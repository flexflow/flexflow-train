#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_OUTPUT_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_OUTPUT_QUERY_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/slot_value_reference.h"

namespace FlexFlow {

template <typename SlotName>
KwargDataflowOutputQuery<SlotName> kwarg_dataflow_output_query_all() {
  return KwargDataflowOutputQuery<SlotName> {
    /*nodes=*/query_set<Node>::matchall(),
    /*output_idxs=*/query_set<SlotName>::matchall(),
  };
}

template <typename SlotName>
bool kwarg_dataflow_output_query_includes(
  KwargDataflowOutputQuery<SlotName> const &query,
  KwargDataflowOutput<SlotName> const &output)
{
  return includes(query.nodes, output.node)
    && includes(query.output_idxs, get_slot_name_for_slot_value_reference(output.value_ref));
}

} // namespace FlexFlow

#endif
