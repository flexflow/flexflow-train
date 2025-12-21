#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_KWARG_DATAFLOW_GRAPH_TRANSITIVE_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_KWARG_DATAFLOW_GRAPH_TRANSITIVE_REDUCTION_H

#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/transitive_reduced_kwarg_dataflow_graph_view.dtg.h"

namespace FlexFlow {

template <typename SlotName>
TransitiveReducedKwargDataflowGraphView<SlotName>
  get_kwarg_dataflow_graph_transitive_reduction(KwargDataflowGraphView<SlotName> const &g) {

  DiGraphView as_digraph = g;
  DiGraphView transitive_reduced = transitive_reduction(as_digraph);

  return TransitiveReducedKwargDataflowGraphView<SlotName>{
      /*full_dataflow_graph=*/g,
      /*transitive_reduction=*/transitive_reduced,
  };
}

} // namespace FlexFlow

#endif
