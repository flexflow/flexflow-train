#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/shape_inference.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/get_dataflow_edges_from_node_to_node.h"
#include "utils/graph/dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/graph/dataflow_graph/dataflow_edge.dtg.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_subgraph_successors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/as_dot.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node.dtg.h"
#include "utils/record_formatter.h"
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

ParallelLayerAddedResult add_parallel_layer(
    ParallelComputationGraph &pcg,
    ParallelLayerAttrs const &layer_attrs,
    std::vector<parallel_tensor_guid_t> const &inputs,
    std::vector<parallel_tensor_guid_t> const &weights,
    std::optional<std::vector<CreateGrad>> const &maybe_output_flags) {
  std::vector<ParallelTensorShape> input_shapes =
      transform(inputs, [&](parallel_tensor_guid_t const &i) {
        return get_parallel_tensor_shape(pcg, i);
      });

  std::vector<ParallelTensorShape> weight_shapes =
      transform(weights, [&](parallel_tensor_guid_t const &i) {
        return get_parallel_tensor_shape(pcg, i);
      });

  std::vector<ParallelTensorShape> correct_weight_shapes =
      get_weight_shapes(layer_attrs.op_attrs, input_shapes);

  if (weight_shapes != correct_weight_shapes) {
    throw mk_runtime_error(
        fmt::format("add_parallel_layer expected weight shapes {}, but "
                    "received weights with shapes {}",
                    correct_weight_shapes,
                    weight_shapes));
  }

  std::vector<ParallelTensorShape> output_shapes =
      get_output_shapes(layer_attrs.op_attrs, input_shapes);

  std::vector<DataflowOutput> unwrapped_inputs =
      transform(inputs, [](parallel_tensor_guid_t const &t) {
        return t.raw_graph_output;
      });

  std::vector<DataflowOutput> unwrapped_weights =
      transform(weights, [](parallel_tensor_guid_t const &t) {
        return t.raw_graph_output;
      });

  std::vector<CreateGrad> output_flags = maybe_output_flags.value_or(
      repeat_element(num_elements(output_shapes), CreateGrad::YES));

  std::vector<ParallelTensorAttrs> output_attrs = zip_with_strict(
      output_shapes,
      output_flags,
      [](ParallelTensorShape const &shape, CreateGrad const &create_grad) {
        return ParallelTensorAttrs{shape, create_grad};
      });

  NodeAddedResult op_added = pcg.raw_graph.add_node(
      layer_attrs,
      concat_vectors(unwrapped_inputs, unwrapped_weights),
      output_attrs);

  return ParallelLayerAddedResult{
      parallel_layer_guid_t{op_added.node},
      transform(
          op_added.outputs,
          [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; }),
  };
}

ParallelLayerAddedResult pcg_add_input_layer(ParallelComputationGraph &pcg,
                                             TensorShape const &tensor_shape) {
  ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{InputAttrs{tensor_shape}},
      /*name=*/std::nullopt,
  };

  return add_parallel_layer(/*pcg=*/pcg,
                            /*layer_attrs=*/layer_attrs,
                            /*inputs=*/{},
                            /*weights=*/{},
                            /*output_flags=*/std::vector{CreateGrad::NO});
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

std::unordered_set<parallel_layer_guid_t>
    get_successors(ParallelComputationGraph const &pcg,
                   parallel_layer_guid_t const &l) {
  return transform(get_successors(pcg.raw_graph, l.raw_graph_node),
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
}

std::unordered_set<parallel_layer_guid_t> get_subgraph_successors(
    ParallelComputationGraph const &pcg,
    std::unordered_set<parallel_layer_guid_t> const &subgraph_layers) {

  std::unordered_set<Node> raw_subgraph_nodes =
      transform(subgraph_layers, [](parallel_layer_guid_t const &l) {
        return l.raw_graph_node;
      });
  std::unordered_set<Node> raw_successors =
      get_subgraph_successors(pcg.raw_graph, raw_subgraph_nodes);

  return transform(raw_successors,
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
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

std::string as_dot(ParallelComputationGraph const &cg) {
  std::function<std::string(ParallelLayerAttrs const &)> get_node_label =
      [](ParallelLayerAttrs const &a) -> std::string {
    RecordFormatter r = as_dot(a.op_attrs);

    if (a.name.has_value()) {
      RecordFormatter rr;
      rr << "Name" << a.name.value();
      r << rr;
    }

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  std::function<std::string(ParallelTensorAttrs const &)> get_input_label =
      [](ParallelTensorAttrs const &a) -> std::string {
    RecordFormatter r;

    r << fmt::to_string(a.shape);

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  return as_dot(view_as_labelled_open_dataflow_graph(cg.raw_graph),
                get_node_label,
                get_input_label);
}

void debug_print_dot(ParallelComputationGraph const &cg) {
  std::cout << as_dot(cg) << std::endl;
}

} // namespace FlexFlow
