#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/get_dataflow_edges_from_node_to_node.h"
#include "utils/graph/dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/graph/dataflow_graph/dataflow_edge.dtg.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node.dtg.h"
#include <unordered_set>

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph() {
  return ParallelComputationGraph{
      LabelledDataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>::create<
          UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs,
                                                ParallelTensorAttrs>>()};
}

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(ParallelComputationGraph const &pcg) {
  return transform(get_nodes(pcg.raw_graph),
                   [&](Node const &n) { return parallel_layer_guid_t{n}; });
}

ParallelLayerAddedResult
    add_parallel_layer(ParallelComputationGraph &pcg,
                       ParallelLayerAttrs const &layer_attrs,
                       std::vector<parallel_tensor_guid_t> const &inputs,
                       std::vector<ParallelTensorAttrs> const &output_labels) {
  std::vector<DataflowOutput> unwrapped_inputs =
      transform(inputs, [](parallel_tensor_guid_t const &t) {
        return t.raw_graph_output;
      });
  NodeAddedResult op_added =
      pcg.raw_graph.add_node(layer_attrs, unwrapped_inputs, output_labels);
  return ParallelLayerAddedResult{
      parallel_layer_guid_t{op_added.node},
      transform(
          op_added.outputs,
          [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; }),
  };
}

ParallelLayerAddedResult
    pcg_add_input_layer(ParallelComputationGraph &pcg,
                        ParallelTensorShape const &tensor_shape) {
  ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{InputAttrs{}},
      /*name=*/std::nullopt,
  };

  ParallelTensorAttrs tensor_attrs = ParallelTensorAttrs{
      /*shape=*/tensor_shape,
      /*sync_type=*/std::nullopt,
      /*initializer=*/std::nullopt,
      /*create_gradients=*/CreateGrad::NO,
  };

  return add_parallel_layer(/*pcg=*/pcg,
                            /*layer_attrs=*/layer_attrs,
                            /*inputs=*/{},
                            /*output_labels=*/{tensor_attrs});
}

std::unordered_set<ParallelComputationGraphEdge>
    get_edges(ParallelComputationGraph const &pcg) {
  return transform(get_edges(pcg.raw_graph), [](DataflowEdge const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::unordered_set<ParallelComputationGraphEdge>
    get_pcg_edges_from_layer_to_layer(ParallelComputationGraph const &pcg,
                                      parallel_layer_guid_t const &src,
                                      parallel_layer_guid_t const &dst) {
  std::unordered_set<DataflowEdge> raw_edges =
      get_dataflow_edges_from_node_to_node(
          pcg.raw_graph, src.raw_graph_node, dst.raw_graph_node);
  return transform(raw_edges, [](DataflowEdge const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::unordered_set<ParallelComputationGraphEdge>
    get_outgoing_edges(ParallelComputationGraph const &pcg,
                       parallel_layer_guid_t const &l) {
  std::unordered_set<DataflowEdge> raw_edges =
      get_outgoing_edges(pcg.raw_graph, l.raw_graph_node);
  return transform(raw_edges, [](DataflowEdge const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::unordered_set<ParallelComputationGraphEdge>
    get_incoming_edges(ParallelComputationGraph const &pcg,
                       parallel_layer_guid_t const &l) {
  std::unordered_set<DataflowEdge> raw_edges =
      unordered_set_of(get_incoming_edges(pcg.raw_graph, l.raw_graph_node));
  return transform(raw_edges, [](DataflowEdge const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::unordered_set<parallel_layer_guid_t>
    get_initial_layers(ParallelComputationGraph const &pcg) {
  std::unordered_set<Node> raw_sources = get_initial_nodes(pcg.raw_graph);
  return transform(raw_sources,
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
}

std::vector<parallel_tensor_guid_t>
    get_incoming_tensors(ParallelComputationGraph const &pcg,
                         parallel_layer_guid_t const &l) {
  return transform(
      get_input_values(pcg.raw_graph, l.raw_graph_node),
      [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; });
}

std::vector<parallel_tensor_guid_t>
    get_layer_outputs(ParallelComputationGraph const &pcg,
                      parallel_layer_guid_t const &l) {
  return transform(
      get_outputs(pcg.raw_graph, l.raw_graph_node),
      [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; });
}

static std::vector<parallel_tensor_guid_t>
    get_incoming_tensors_with_role(ParallelComputationGraph const &pcg,
                                   parallel_layer_guid_t const &l,
                                   IncomingTensorRole desired_role) {
  PCGOperatorAttrs attrs = get_parallel_layer_attrs(pcg, l).op_attrs;

  std::vector<parallel_tensor_guid_t> incoming_tensors =
      get_incoming_tensors(pcg, l);

  std::vector<IncomingTensorRole> incoming_tensor_roles =
      get_incoming_tensor_roles(attrs, incoming_tensors.size());

  assert(incoming_tensors.size() == incoming_tensor_roles.size());

  std::vector<parallel_tensor_guid_t> result = filtrans(
      zip(incoming_tensors, incoming_tensor_roles),
      [&](std::pair<parallel_tensor_guid_t, IncomingTensorRole> const &p)
          -> std::optional<parallel_tensor_guid_t> {
        parallel_tensor_guid_t tensor = p.first;
        IncomingTensorRole role = p.second;

        if (role == desired_role) {
          return tensor;
        } else {
          return std::nullopt;
        }
      });
  return result;
}

std::vector<parallel_tensor_guid_t>
    get_incoming_inputs(ParallelComputationGraph const &pcg,
                        parallel_layer_guid_t const &l) {
  return get_incoming_tensors_with_role(pcg, l, IncomingTensorRole::INPUT);
}

std::vector<parallel_tensor_guid_t>
    get_incoming_weights(ParallelComputationGraph const &pcg,
                         parallel_layer_guid_t const &l) {
  return get_incoming_tensors_with_role(pcg, l, IncomingTensorRole::WEIGHT);
}

parallel_layer_guid_t get_source_layer(ParallelComputationGraph const &g,
                                       parallel_tensor_guid_t const &t) {
  return parallel_layer_guid_t{t.raw_graph_output.node};
}

ParallelLayerAttrs get_parallel_layer_attrs(ParallelComputationGraph const &pcg,
                                            parallel_layer_guid_t const &l) {
  return pcg.raw_graph.at(l.raw_graph_node);
}

PCGOperatorAttrs pcg_get_op_attrs(ParallelComputationGraph const &pcg,
                                  parallel_layer_guid_t const &l) {
  return get_parallel_layer_attrs(pcg, l).op_attrs;
}

ParallelTensorAttrs
    get_parallel_tensor_attrs(ParallelComputationGraph const &pcg,
                              parallel_tensor_guid_t const &t) {
  return pcg.raw_graph.at(t.raw_graph_output);
}

ParallelTensorShape
    get_parallel_tensor_shape(ParallelComputationGraph const &pcg,
                              parallel_tensor_guid_t const &t) {
  return get_parallel_tensor_attrs(pcg, t).shape;
}

std::vector<parallel_layer_guid_t>
    topological_ordering(ParallelComputationGraph const &pcg) {
  return transform(get_topological_ordering(pcg.raw_graph),
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
}

parallel_layer_guid_t
    get_parallel_layer_by_name(ParallelComputationGraph const &pcg,
                               std::string const &name) {
  std::unordered_set<parallel_layer_guid_t> found =
      filter(get_parallel_layers(pcg), [&](parallel_layer_guid_t const &l) {
        return get_parallel_layer_attrs(pcg, l).name == name;
      });
  return get_only(found);
}

ParallelComputationGraph
    without_layer_names(ParallelComputationGraph const &pcg) {
  return ParallelComputationGraph{
      LabelledDataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>::
          create_copy_of<
              UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs,
                                                    ParallelTensorAttrs>>(
              rewrite_node_labels(
                  pcg.raw_graph,
                  [](Node const &n, ParallelLayerAttrs const &old_attrs) {
                    ParallelLayerAttrs new_attrs = old_attrs;
                    new_attrs.name = std::nullopt;
                    return new_attrs;
                  })),
  };
}

bool pcgs_are_isomorphic(ParallelComputationGraph const &lhs,
                         ParallelComputationGraph const &rhs) {
  return find_isomorphism(without_layer_names(lhs).raw_graph,
                          without_layer_names(rhs).raw_graph)
      .has_value();
}

} // namespace FlexFlow
