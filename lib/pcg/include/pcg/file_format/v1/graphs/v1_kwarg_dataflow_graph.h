#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_KWARG_DATAFLOW_GRAPH_H

#include "pcg/file_format/v1/graphs/v1_graph_edge.dtg.h"
#include "pcg/file_format/v1/graphs/v1_kwarg_dataflow_graph.dtg.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"
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

template <typename SlotName>
std::pair<KwargDataflowGraphView<SlotName>,
          std::unordered_map<nonnegative_int, Node>>
    from_v1_including_node_numbering(V1KwargDataflowGraph<SlotName> const &v1) {
  std::unordered_map<nonnegative_int, Node> node_map =
      generate_map(v1.nodes, [](nonnegative_int n) {
        return Node{n.size_t_from_nonnegative_int()};
      });
  std::unordered_set<Node> node_set = unordered_set_of(values(node_map));

  std::unordered_set<OpenKwargDataflowEdge<int, SlotName>> edges =
      transform(v1.edges, [](V1GraphEdge<SlotName> const &e) {
        Node srcNode = Node{e.srcNode.size_t_from_nonnegative_int()};
        Node dstNode = Node{e.dstNode.size_t_from_nonnegative_int()};
        return OpenKwargDataflowEdge<int, SlotName>{KwargDataflowEdge<SlotName>{
            /*src=*/KwargDataflowOutput<SlotName>{srcNode, e.srcSlot},
            /*dst=*/KwargDataflowInput<SlotName>{dstNode, e.dstSlot},
        }};
      });

  OpenKwargDataflowGraphData<int, SlotName> graph_data =
      OpenKwargDataflowGraphData<int, SlotName>{
          /*nodes=*/node_set,
          /*edges=*/edges,
          /*inputs=*/{},
          /*outputs=*/{},
      };
  return std::pair{view_from_open_kwarg_dataflow_graph_data(graph_data),
                   node_map};
}

template <typename SlotName>
KwargDataflowGraphView<SlotName>
    from_v1(V1KwargDataflowGraph<SlotName> const &v1) {
  return from_v1_including_node_numbering(v1).first;
}

} // namespace FlexFlow

#endif
