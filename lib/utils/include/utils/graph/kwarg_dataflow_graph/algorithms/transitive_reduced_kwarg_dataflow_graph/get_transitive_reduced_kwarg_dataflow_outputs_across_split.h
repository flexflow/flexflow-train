#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_TRANSITIVE_REDUCED_KWARG_DATAFLOW_OUTPUTS_ACROSS_SPLIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_TRANSITIVE_REDUCED_KWARG_DATAFLOW_OUTPUTS_ACROSS_SPLIT_H

#include "utils/containers/transform.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/get_transitive_reduced_kwarg_dataflow_edges_across_split.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/transitive_reduced_kwarg_dataflow_graph_view.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.dtg.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowOutput<SlotName>>
    get_transitive_reduced_kwarg_dataflow_outputs_across_split(
        TransitiveReducedKwargDataflowGraphView<SlotName> const &tr_g,
        BinarySeriesSplit const &split) {

  return transform(
      get_transitive_reduced_kwarg_dataflow_edges_across_split(tr_g, split),
      [](KwargDataflowEdge<SlotName> const &e) { return e.src; });
}

} // namespace FlexFlow

#endif
