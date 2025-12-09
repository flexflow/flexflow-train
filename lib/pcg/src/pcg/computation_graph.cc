#include "pcg/computation_graph.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/shape_inference.h"
#include "utils/containers/filter_values.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/get_subgraph_incoming_edges.h"
#include "utils/graph/dataflow_graph/algorithms/get_subgraph_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_subgraph_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_incoming_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_subgraph_incoming_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_subgraph_outgoing_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/rewrite_labelled_kwarg_dataflow_graph_node_labels.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/as_dot.h"
#include "utils/graph/node/algorithms.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph() {
  return ComputationGraph{
      LabelledKwargDataflowGraph<LayerAttrs, TensorAttrs, TensorSlotName>::create<
          UnorderedSetLabelledOpenKwargDataflowGraph<LayerAttrs, TensorAttrs, int, TensorSlotName>>()};
}

std::unordered_set<layer_guid_t> get_layers(ComputationGraph const &cg) {
  return transform(get_nodes(cg.raw_graph),
                   [&](Node const &n) { return layer_guid_t{n}; });
}

LayerAddedResult add_layer(
    ComputationGraph &computation_graph,
    LayerAttrs const &layer_attrs,
    std::unordered_map<TensorSlotName, tensor_guid_t> const &inputs,
    std::unordered_map<TensorSlotName, tensor_guid_t> const &weights,
    std::optional<std::unordered_map<TensorSlotName, CreateGrad>> const &maybe_output_flags) {

  std::unordered_map<TensorSlotName, TensorShape> input_shapes =
      map_values(
        inputs, 
        [&](tensor_guid_t const &i) {
          return get_tensor_attrs(computation_graph, i).shape;
        });

  std::unordered_map<TensorSlotName, TensorShape> provided_weight_shapes =
      map_values(
        weights, 
        [&](tensor_guid_t const &w) {
          return get_tensor_attrs(computation_graph, w).shape;
        });

  std::unordered_map<TensorSlotName, TensorShape> expected_weight_shapes =
      get_weight_shapes(layer_attrs.op_attrs, input_shapes);

  std::unordered_map<TensorSlotName, KwargDataflowOutput<TensorSlotName>> raw_inputs 
    = map_values(
        inputs, 
        [&](tensor_guid_t const &t) {
          return t.raw_graph_output;
        });

  std::unordered_map<TensorSlotName, KwargDataflowOutput<TensorSlotName>> raw_weights = map_values(
      weights, 
      [&](tensor_guid_t const &t) {
        return t.raw_graph_output;
      });
  std::unordered_map<TensorSlotName, TensorShape> output_shapes =
      get_output_shapes(layer_attrs.op_attrs, input_shapes);

  std::unordered_map<TensorSlotName, CreateGrad> output_flags 
    = maybe_output_flags.value_or(
        map_values(
          output_shapes,
          [&](TensorShape const &) {
            return CreateGrad::YES; 
          }));

  std::unordered_map<TensorSlotName, TensorAttrs> output_attrs = 
    zip_values_strict_with(
      output_shapes,
      output_flags,
      [](TensorShape const &shape, CreateGrad const &create_grad) 
        -> TensorAttrs
      {
        return TensorAttrs{
            /*shape=*/shape,
            /*create_grad=*/create_grad,
        };
      });

  KwargNodeAddedResult<TensorSlotName> added = computation_graph.raw_graph.add_node(
      layer_attrs, binary_merge_disjoint_maps(raw_inputs, raw_weights), output_attrs);

  return LayerAddedResult{
      layer_guid_t{added.node},
      map_values(added.outputs,
                [](KwargDataflowOutput<TensorSlotName> const &o) { return tensor_guid_t{o}; }),
  };
}

LayerAddedResult add_input_layer(ComputationGraph &cg,
                                 TensorShape const &tensor_shape) {
  LayerAttrs layer_attrs = LayerAttrs{
      /*op_attrs=*/ComputationGraphOpAttrs{InputAttrs{tensor_shape}},
      /*name=*/std::nullopt,
  };

  return add_layer(cg,
                   layer_attrs,
                   /*inputs=*/{},
                   /*weights=*/{},
                   /*outputs=*/std::unordered_map<TensorSlotName, CreateGrad>{
                     {TensorSlotName::OUTPUT, CreateGrad::NO},
                   });
}

LayerAddedResult add_input_layer_with_grad(ComputationGraph &cg,
                                           TensorShape const &tensor_shape) {
  LayerAttrs layer_attrs = LayerAttrs{
      /*op_attrs=*/ComputationGraphOpAttrs{InputAttrs{tensor_shape}},
      /*name=*/std::nullopt,
  };

  return add_layer(cg,
                   layer_attrs,
                   /*inputs=*/{},
                   /*weights=*/{},
                   /*outputs=*/std::unordered_map<TensorSlotName, CreateGrad>{
                     {TensorSlotName::OUTPUT, CreateGrad::YES},
                   });
}

TensorAttrs get_tensor_attrs(ComputationGraph const &cg,
                             tensor_guid_t const &t) {
  return cg.raw_graph.at(t.raw_graph_output);
}

bool are_tensor_guid_shapes_equivalent(ComputationGraph const &cg,
                                       tensor_guid_t const &t1,
                                       tensor_guid_t const &t2) {
  return get_tensor_attrs(cg, t1).shape == get_tensor_attrs(cg, t2).shape;
}

std::vector<layer_guid_t> topological_ordering(ComputationGraph const &cg) {
  std::vector<Node> layers = get_topological_ordering(cg.raw_graph);
  return transform(
      layers, [&](Node const &e) -> layer_guid_t { return layer_guid_t{e}; });
}

std::vector<layer_guid_t>
    reverse_topological_ordering(ComputationGraph const &cg) {
  std::vector<Node> layers = reversed(get_topological_ordering(cg.raw_graph));
  return transform(
      layers, [&](Node const &e) -> layer_guid_t { return layer_guid_t{e}; });
}

std::unordered_map<TensorSlotName, tensor_guid_t> get_outgoing_tensors(ComputationGraph const &cg,
                                                                       layer_guid_t n) {
  return map_values(get_outgoing_kwarg_dataflow_outputs_for_node(cg.raw_graph, n.raw_node),
                    [](KwargDataflowOutput<TensorSlotName> const &o) { return tensor_guid_t{o}; });
}

std::unordered_map<TensorSlotName, tensor_guid_t> get_incoming_tensors(ComputationGraph const &cg,
                                                                       layer_guid_t n) {
  return map_values(get_incoming_kwarg_dataflow_outputs_for_node(cg.raw_graph, n.raw_node),
                   [](KwargDataflowOutput<TensorSlotName> const &o) { return tensor_guid_t{o}; });
}

std::unordered_map<TensorSlotName, TensorShape> get_incoming_input_shapes(ComputationGraph const &cg,
                                                                          layer_guid_t const &n) {
  return map_values(get_incoming_inputs(cg, n), [&](tensor_guid_t const &t) {
    return get_tensor_attrs(cg, t).shape;
  });
}

static std::unordered_map<TensorSlotName, tensor_guid_t>
    get_incoming_tensors_with_role(ComputationGraph const &cg,
                                   layer_guid_t const &l,
                                   IncomingTensorRole desired_role) {
  ComputationGraphOpAttrs attrs = get_layer_attrs(cg, l).op_attrs;

  std::unordered_map<TensorSlotName, tensor_guid_t> incoming_tensors = get_incoming_tensors(cg, l);

  std::unordered_map<TensorSlotName, IncomingTensorRole> incoming_slot_roles =
      get_incoming_tensor_roles(attrs);

  assert(incoming_tensors.size() == incoming_slot_roles.size());

  std::unordered_set<TensorSlotName> slots_with_desired_role = 
    keys(filter_values(incoming_slot_roles, 
                       [&](IncomingTensorRole role) {
                         return role == desired_role;
                       }));

  return restrict_keys(incoming_tensors, slots_with_desired_role);
}

std::unordered_map<TensorSlotName, tensor_guid_t> get_incoming_inputs(ComputationGraph const &cg,
                                               layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::INPUT);
}

std::unordered_map<TensorSlotName, tensor_guid_t> get_incoming_weights(ComputationGraph const &cg,
                                                layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::WEIGHT);
}

std::unordered_set<tensor_guid_t> get_all_tensors(ComputationGraph const &cg) {
  return transform(get_all_kwarg_dataflow_outputs(cg.raw_graph),
                   [](KwargDataflowOutput<TensorSlotName> const &t) { return tensor_guid_t(t); });
}

std::unordered_map<tensor_guid_t, TensorAttrs>
    get_all_tensor_attrs(ComputationGraph const &cg) {
  std::unordered_set<tensor_guid_t> all_tensors = get_all_tensors(cg);
  std::unordered_map<tensor_guid_t, TensorAttrs> all_tensor_attrs;
  for (tensor_guid_t const &tensor_guid : all_tensors) {
    all_tensor_attrs.insert({tensor_guid, get_tensor_attrs(cg, tensor_guid)});
  }
  return all_tensor_attrs;
}

std::unordered_set<ComputationGraphEdge> get_subgraph_incoming_edges(
    ComputationGraph const &cg,
    std::unordered_set<layer_guid_t> const &subgraph_nodes) {

  std::unordered_set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::unordered_set<KwargDataflowEdge<TensorSlotName>> raw_incoming_edges =
      get_kwarg_dataflow_subgraph_incoming_edges(cg.raw_graph, raw_subgraph_nodes);

  return transform(raw_incoming_edges, [](KwargDataflowEdge<TensorSlotName> const &e) {
    return ComputationGraphEdge{e};
  });
}

std::unordered_set<ComputationGraphEdge> get_subgraph_outgoing_edges(
    ComputationGraph const &cg,
    std::unordered_set<layer_guid_t> const &subgraph_nodes) {

  std::unordered_set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::unordered_set<KwargDataflowEdge<TensorSlotName>> raw_outgoing_edges =
      get_kwarg_dataflow_subgraph_outgoing_edges(cg.raw_graph, raw_subgraph_nodes);

  return transform(raw_outgoing_edges, [](KwargDataflowEdge<TensorSlotName> const &e) {
    return ComputationGraphEdge{e};
  });
}

std::unordered_set<layer_guid_t> get_subgraph_successors(
    ComputationGraph const &cg,
    std::unordered_set<layer_guid_t> const &subgraph_nodes) {

  std::unordered_set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::unordered_set<Node> raw_successors =
      get_subgraph_successors(cg.raw_graph, raw_subgraph_nodes);

  return transform(raw_successors,
                   [](Node const &n) { return layer_guid_t{n}; });
}

LayerAttrs get_layer_attrs(ComputationGraph const &cg, layer_guid_t const &n) {
  return cg.raw_graph.at(n.raw_node);
}

std::unordered_map<layer_guid_t, LayerAttrs>
    get_layer_attrs_mapping(ComputationGraph const &cg) {
  std::unordered_map<layer_guid_t, LayerAttrs> layer_attrs_mapping;
  for (layer_guid_t const &layer_guid : get_layers(cg)) {
    layer_attrs_mapping.insert({layer_guid, get_layer_attrs(cg, layer_guid)});
  }
  return layer_attrs_mapping;
}

layer_guid_t get_layer_by_name(ComputationGraph const &cg,
                               std::string const &name) {
  std::unordered_set<layer_guid_t> found =
      filter(get_layers(cg), [&](layer_guid_t const &l) {
        return get_layer_attrs(cg, l).name == name;
      });
  return get_only(found);
}

ComputationGraph without_layer_names(ComputationGraph const &cg) {
  return ComputationGraph{
      LabelledKwargDataflowGraph<LayerAttrs, TensorAttrs, TensorSlotName>::create_copy_of<
          UnorderedSetLabelledOpenKwargDataflowGraph<LayerAttrs, TensorAttrs, int, TensorSlotName>>(
            rewrite_labelled_kwarg_dataflow_graph_node_labels(cg.raw_graph,
                                [](Node const &n, LayerAttrs const &old_attrs) {
                                  LayerAttrs new_attrs = old_attrs;
                                  new_attrs.name = std::nullopt;
                                  return new_attrs;
                                })),
  };
}

bool computation_graphs_are_isomorphic(ComputationGraph const &lhs,
                                       ComputationGraph const &rhs) {
  return find_isomorphism(without_layer_names(lhs).raw_graph,
                          without_layer_names(rhs).raw_graph)
      .has_value();
}

std::string as_dot(ComputationGraph const &cg) {
  std::function<std::string(LayerAttrs const &)> get_node_label =
      [](LayerAttrs const &a) -> std::string {
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

  std::function<std::string(TensorAttrs const &)> get_input_label =
      [](TensorAttrs const &a) -> std::string {
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

void debug_print_dot(ComputationGraph const &cg) {
  std::cout << as_dot(cg) << std::endl;
}

} // namespace FlexFlow
