#include "compiler/task_graph_simulator/pcg_task_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/graph/instances/adjacency_digraph.h"

namespace FlexFlow {

PCGTaskGraph get_pcg_task_graph(ParallelComputationGraph const &pcg) {
  DiGraph digraph = DiGraph::create<AdjacencyDiGraph>();
  bidict<Node, PCGTask> node_map;

  for (parallel_layer_guid_t const &layer : get_parallel_layers(pcg)) {
    node_map.equate(digraph.add_node(), PCGTask{layer});
  }

  for (ParallelComputationGraphEdge const &edge : get_edges(pcg)) {
    node_map.equate(digraph.add_node(), PCGTask{edge});
  }

  for (auto const &[node, task] : node_map.as_unordered_map()) {
    if (task.is_edge()) {
      ParallelComputationGraphEdge edge = task.require_edge();
      parallel_layer_guid_t src_layer = get_src_layer(edge);
      parallel_layer_guid_t dst_layer = get_dst_layer(edge);

      Node src_node = node_map.at_r(PCGTask{src_layer});
      Node dst_node = node_map.at_r(PCGTask{dst_layer});

      digraph.add_edge(DirectedEdge{src_node, node});
      digraph.add_edge(DirectedEdge{node, dst_node});
    }
  }
  return PCGTaskGraph{digraph, node_map};
}

} // namespace FlexFlow
