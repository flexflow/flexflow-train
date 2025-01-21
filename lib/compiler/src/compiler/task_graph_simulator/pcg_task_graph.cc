#include "compiler/task_graph_simulator/pcg_task_graph.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/tensor_set_movement.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <unordered_map>

namespace FlexFlow {

PCGTaskGraph get_pcg_task_graph(ParallelComputationGraph const &pcg,
                                MachineMapping const &machine_mapping,
                                MachineSpecification const &machine_spec) {
  DiGraph digraph = DiGraph::create<AdjacencyDiGraph>();
  bidict<Node, PCGTask> node_to_task;
  bidict<Node, parallel_layer_guid_t> node_to_layer;

  for (parallel_layer_guid_t const &layer : get_parallel_layers(pcg)) {
    MachineView mv = machine_mapping.machine_views.at(layer);
    OpCostEstimateKey op_key =
        get_mapped_op_cost_estimate_key_for_layer(pcg, layer, mv);
    Node n = digraph.add_node();
    node_to_task.equate(n, PCGTask{op_key});
    node_to_layer.equate(n, layer);
  }

  for (ParallelComputationGraphEdge const &edge : get_edges(pcg)) {
    MachineView src_mv = machine_mapping.machine_views.at(get_src_layer(edge));
    MachineView dst_mv = machine_mapping.machine_views.at(get_dst_layer(edge));
    TensorSetMovement movement =
        get_tensor_set_movement_from_pcg_edge(edge, pcg, src_mv, dst_mv);
    Node node = digraph.add_node();
    node_to_task.equate(node, PCGTask{movement});
    Node src_node = node_to_layer.at_r(get_src_layer(edge));
    Node dst_node = node_to_layer.at_r(get_dst_layer(edge));

    digraph.add_edge(DirectedEdge{src_node, node});
    digraph.add_edge(DirectedEdge{node, dst_node});
  }

  return PCGTaskGraph{digraph, node_to_task, node_to_layer};
}
} // namespace FlexFlow
