#include "compiler/series_parallel/pcg/get_pcg_series_parallel_decomposition.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"

namespace FlexFlow {

std::optional<SeriesParallelDecomposition>
    get_pcg_series_parallel_decomposition(ParallelComputationGraph const &pcg) {
  {
    DiGraphView unpreprocessed_digraph = pcg.raw_graph;
    std::optional<SeriesParallelDecomposition> unpreprocessed_sp_decomposition =
        get_series_parallel_decomposition(unpreprocessed_digraph);
    if (unpreprocessed_sp_decomposition.has_value()) {
      return unpreprocessed_sp_decomposition.value();
    }
  }

  auto layer_is_weight_or_input = [&](parallel_layer_guid_t const &l) {
    PCGOperatorAttrs op_attrs = get_parallel_layer_attrs(pcg, l).op_attrs;
    return op_attrs.has<WeightAttrs>() || op_attrs.has<InputAttrs>();
  };

  auto layer_is_parallel_op = [&](parallel_layer_guid_t const &l) {
    PCGOperatorAttrs op_attrs = get_parallel_layer_attrs(pcg, l).op_attrs;
    return is_parallel_op(op_attrs);
  };

  std::function<parallel_layer_guid_t(parallel_layer_guid_t const &)>
      follow_to_last_parallel_op =
          [&](parallel_layer_guid_t const &starting_point)
      -> parallel_layer_guid_t {
    assert(layer_is_weight_or_input(starting_point) ||
           layer_is_parallel_op(starting_point));

    std::unordered_set<parallel_layer_guid_t> successors =
        get_successors(pcg, starting_point);

    if (successors.size() != 1) {
      return starting_point;
    }

    parallel_layer_guid_t successor =
        get_only(get_successors(pcg, starting_point));

    assert(!layer_is_weight_or_input(successor));
    if (layer_is_parallel_op(successor)) {
      return follow_to_last_parallel_op(successor);
    } else {
      return starting_point;
    }
  };

  DiGraphView preprocessed_digraph = [&] {
    std::unordered_set<parallel_layer_guid_t> weight_and_input_layers =
        filter(get_parallel_layers(pcg), layer_is_weight_or_input);

    std::unordered_set<parallel_layer_guid_t> par_chain_endpoints =
        transform(weight_and_input_layers, follow_to_last_parallel_op);

    std::unordered_set<parallel_layer_guid_t> par_chain_endpoint_successors =
        get_subgraph_successors(pcg, par_chain_endpoints);

    DiGraph digraph = materialize_digraph_view<AdjacencyDiGraph>(pcg.raw_graph);
    for (parallel_layer_guid_t const &src : par_chain_endpoints) {
      for (parallel_layer_guid_t const &dst : par_chain_endpoint_successors) {
        digraph.add_edge(DirectedEdge{src.raw_graph_node, dst.raw_graph_node});
      }
    }

    return digraph;
  }();

  return get_series_parallel_decomposition(preprocessed_digraph);
}

} // namespace FlexFlow
