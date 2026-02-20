#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_KWARG_DATAFLOW_GRAPH_H

#include "pcg/file_format/v1/graphs/v1_kwarg_dataflow_graph.dtg.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/sorted.h"
#include "utils/containers/values.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

template <typename SlotName>
V1KwargDataflowGraph<SlotName>
    to_v1(KwargDataflowGraphView<SlotName> const &g) {
  bidict<nonnegative_int, Node> node_enumeration_bidict =
      bidict_from_enumerating(get_nodes(g));
  std::unordered_map<Node, nonnegative_int> node_enumeration =
      node_enumeration_bidict.reversed().as_unordered_map();
  return to_v1(g, node_enumeration);
}

template <typename SlotName>
V1KwargDataflowGraph<SlotName>
    to_v1(KwargDataflowGraphView<SlotName> const &g,
          std::unordered_map<Node, nonnegative_int> const &nodes) {
  std::unordered_set<V1GraphEdge<SlotName>> edges;
  for (KwargDataflowEdge<SlotName> const &e : get_all_kwarg_dataflow_edges(g)) {
    edges.insert(V1GraphEdge{nodes.at(e.src.node),
                             e.src.slot_name,
                             nodes.at(e.dst.node),
                             e.dst.slot_name});
  }

  return V1KwargDataflowGraph<SlotName>{
      sorted(values(nodes)),
      edges,
  };
}

} // namespace FlexFlow

#endif
