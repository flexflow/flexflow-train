#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/shape_inference.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_subgraph_successors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/find_isomorphism_between_kwarg_dataflow_graphs.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_incoming_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_edges_from_node_to_node.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_edges_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_view_as_dot.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/rewrite_labelled_kwarg_dataflow_graph_node_labels.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node.dtg.h"
#include "utils/record_formatter.h"
#include <unordered_set>

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph() {
  return ParallelComputationGraph{
      LabelledKwargDataflowGraph<ParallelLayerAttrs,
                                 ParallelTensorAttrs,
                                 TensorSlotName>::
          create<UnorderedSetLabelledOpenKwargDataflowGraph<ParallelLayerAttrs,
                                                            ParallelTensorAttrs,
                                                            int,
                                                            TensorSlotName>>()};
}

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(ParallelComputationGraph const &pcg) {
  return transform(get_nodes(pcg.raw_graph),
                   [&](Node const &n) { return parallel_layer_guid_t{n}; });
}

ParallelLayerAddedResult add_parallel_layer(
    ParallelComputationGraph &pcg,
    ParallelLayerAttrs const &layer_attrs,
    std::unordered_map<TensorSlotName, parallel_tensor_guid_t> const &inputs,
    std::unordered_map<TensorSlotName, parallel_tensor_guid_t> const &weights,
    std::optional<std::unordered_map<TensorSlotName, CreateGrad>> const
        &maybe_output_flags) {
  std::unordered_map<TensorSlotName, ParallelTensorShape> input_shapes =
      map_values(inputs, [&](parallel_tensor_guid_t const &i) {
        return get_parallel_tensor_shape(pcg, i);
      });

  std::unordered_map<TensorSlotName, ParallelTensorShape> weight_shapes =
      map_values(weights, [&](parallel_tensor_guid_t const &i) {
        return get_parallel_tensor_shape(pcg, i);
      });

  std::unordered_map<TensorSlotName, ParallelTensorShape>
      correct_weight_shapes =
          get_weight_shapes(layer_attrs.op_attrs, input_shapes);

  ASSERT(weight_shapes == correct_weight_shapes,
         "add_parallel_layer received incorrect weight shapes");

  std::unordered_map<TensorSlotName, ParallelTensorShape> output_shapes =
      get_output_shapes(layer_attrs.op_attrs, input_shapes);

  std::unordered_map<TensorSlotName, KwargDataflowOutput<TensorSlotName>>
      unwrapped_inputs =
          map_values(inputs, [](parallel_tensor_guid_t const &t) {
            return t.raw_graph_output;
          });

  std::unordered_map<TensorSlotName, KwargDataflowOutput<TensorSlotName>>
      unwrapped_weights =
          map_values(weights, [](parallel_tensor_guid_t const &t) {
            return t.raw_graph_output;
          });

  std::unordered_map<TensorSlotName, CreateGrad> output_flags =
      maybe_output_flags.value_or(
          generate_map(keys(output_shapes),
                       [](TensorSlotName const &) { return CreateGrad::YES; }));

  std::unordered_map<TensorSlotName, ParallelTensorAttrs> output_attrs =
      zip_values_strict_with(
          output_shapes,
          output_flags,
          [](ParallelTensorShape const &shape, CreateGrad const &create_grad) {
            return ParallelTensorAttrs{shape, create_grad};
          });

  KwargNodeAddedResult<TensorSlotName> op_added = pcg.raw_graph.add_node(
      layer_attrs,
      binary_merge_disjoint_maps(unwrapped_inputs, unwrapped_weights),
      output_attrs);

  return ParallelLayerAddedResult{
      parallel_layer_guid_t{op_added.node},
      map_values(op_added.outputs,
                 [](KwargDataflowOutput<TensorSlotName> const &o) {
                   return parallel_tensor_guid_t{o};
                 }),
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
                            /*output_flags=*/
                            std::unordered_map<TensorSlotName, CreateGrad>{
                                {
                                    TensorSlotName::OUTPUT,
                                    CreateGrad::NO,
                                },
                            });
}

OperatorTaskSpace get_operator_task_space(ParallelComputationGraph const &pcg,
                                          parallel_layer_guid_t const &layer) {
  PCGOperatorAttrs op_attrs = pcg_get_op_attrs(pcg, layer);

  ASSERT(!is_parallel_op(op_attrs));

  std::unordered_map<TensorSlotName, parallel_tensor_guid_t> inputs =
      get_incoming_inputs(pcg, layer);

  std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> input_degrees =
      map_values(get_incoming_inputs(pcg, layer),
                 [&](parallel_tensor_guid_t input_guid) {
                   return get_parallel_degrees(
                       get_parallel_tensor_shape(pcg, input_guid));
                 });

  return get_operator_task_space(
      compgraph_op_attrs_from_pcg_op_attrs(op_attrs).value(), input_degrees);
}

std::unordered_set<ParallelComputationGraphEdge>
    get_edges(ParallelComputationGraph const &pcg) {
  return transform(get_all_kwarg_dataflow_edges(pcg.raw_graph),
                   [](KwargDataflowEdge<TensorSlotName> const &e) {
                     return ParallelComputationGraphEdge{e};
                   });
}

std::unordered_set<ParallelComputationGraphEdge>
    get_pcg_edges_from_layer_to_layer(ParallelComputationGraph const &pcg,
                                      parallel_layer_guid_t const &src,
                                      parallel_layer_guid_t const &dst) {
  std::unordered_set<KwargDataflowEdge<TensorSlotName>> raw_edges =
      get_kwarg_dataflow_edges_from_node_to_node(
          pcg.raw_graph, src.raw_graph_node, dst.raw_graph_node);
  return transform(raw_edges, [](KwargDataflowEdge<TensorSlotName> const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::unordered_set<ParallelComputationGraphEdge>
    get_outgoing_edges(ParallelComputationGraph const &pcg,
                       parallel_layer_guid_t const &l) {
  std::unordered_set<KwargDataflowEdge<TensorSlotName>> raw_edges =
      get_outgoing_kwarg_dataflow_edges_for_node(pcg.raw_graph,
                                                 l.raw_graph_node)
          .right_values();
  return transform(raw_edges, [](KwargDataflowEdge<TensorSlotName> const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::unordered_map<TensorSlotName, ParallelComputationGraphEdge>
    get_incoming_edges(ParallelComputationGraph const &pcg,
                       parallel_layer_guid_t const &l) {
  std::unordered_map<TensorSlotName, KwargDataflowEdge<TensorSlotName>>
      raw_edges = get_incoming_kwarg_dataflow_edges_for_node(pcg.raw_graph,
                                                             l.raw_graph_node);
  return map_values(raw_edges, [](KwargDataflowEdge<TensorSlotName> const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::unordered_set<parallel_layer_guid_t>
    get_initial_layers(ParallelComputationGraph const &pcg) {
  std::unordered_set<Node> raw_sources = get_initial_nodes(pcg.raw_graph);
  return transform(raw_sources,
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
}

std::unordered_map<TensorSlotName, parallel_tensor_guid_t>
    get_outgoing_tensors(ParallelComputationGraph const &pcg,
                         parallel_layer_guid_t const &l) {
  return map_values(get_outgoing_kwarg_dataflow_outputs_for_node(
                        pcg.raw_graph, l.raw_graph_node),
                    [](KwargDataflowOutput<TensorSlotName> const &o) {
                      return parallel_tensor_guid_t{o};
                    });
}

std::unordered_map<TensorSlotName, parallel_tensor_guid_t>
    get_incoming_tensors(ParallelComputationGraph const &pcg,
                         parallel_layer_guid_t const &l) {
  return map_values(get_incoming_kwarg_dataflow_outputs_for_node(
                        pcg.raw_graph, l.raw_graph_node),
                    [](KwargDataflowOutput<TensorSlotName> const &o) {
                      return parallel_tensor_guid_t{o};
                    });
}

std::unordered_map<TensorSlotName, parallel_tensor_guid_t>
    get_layer_outputs(ParallelComputationGraph const &pcg,
                      parallel_layer_guid_t const &l) {
  return map_values(get_outgoing_kwarg_dataflow_outputs_for_node(
                        pcg.raw_graph, l.raw_graph_node),
                    [](KwargDataflowOutput<TensorSlotName> const &o) {
                      return parallel_tensor_guid_t{o};
                    });
}

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    pcg_get_operator_to_incoming_mappings(ParallelComputationGraph const &pcg,
                                          parallel_layer_guid_t const &l) {
  ComputationGraphOpAttrs op_attrs =
      compgraph_op_attrs_from_pcg_op_attrs(pcg_get_op_attrs(pcg, l)).value();

  return get_operator_to_incoming_mappings(
      /*attrs=*/op_attrs,
      /*input_degrees=*/get_incoming_input_degrees(pcg, l));
}

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    pcg_get_operator_to_output_mappings(ParallelComputationGraph const &pcg,
                                        parallel_layer_guid_t const &l) {
  ComputationGraphOpAttrs op_attrs =
      compgraph_op_attrs_from_pcg_op_attrs(pcg_get_op_attrs(pcg, l)).value();

  return get_operator_to_output_mappings(
      /*attrs=*/op_attrs,
      /*input_degrees=*/get_incoming_input_degrees(pcg, l));
}

OperatorTaskSpaceToOperatorTaskSpaceMapping
    pcg_get_mapping_along_edge(ParallelComputationGraph const &pcg,
                               ParallelComputationGraphEdge const &edge) {

  parallel_layer_guid_t src_layer = get_src_layer(edge);
  TensorSlotName src_slot_name = get_src_layer_output_slot_name(edge);
  parallel_tensor_guid_t tensor = parallel_tensor_guid_t{edge.raw_edge.src};
  parallel_layer_guid_t dst_layer = get_dst_layer(edge);
  TensorSlotName dst_slot_name = get_dst_layer_input_slot_name(edge);

  ParallelTensorShape tensor_shape = get_parallel_tensor_shape(pcg, tensor);

  OperatorTaskSpace src_task_space = get_operator_task_space(pcg, src_layer);

  OperatorTaskSpace dst_task_space = get_operator_task_space(pcg, dst_layer);

  OperatorSpaceToParallelTensorSpaceMapping src_to_tensor_mapping =
      pcg_get_operator_to_output_mappings(pcg, src_layer).at(src_slot_name);

  OperatorSpaceToParallelTensorSpaceMapping dst_to_tensor_mapping =
      pcg_get_operator_to_incoming_mappings(pcg, dst_layer).at(dst_slot_name);

  return op_to_op_mapping_from_composition_through_tensor(
      src_to_tensor_mapping, dst_to_tensor_mapping);
}

static std::unordered_map<TensorSlotName, parallel_tensor_guid_t>
    get_incoming_tensors_with_role(ParallelComputationGraph const &pcg,
                                   parallel_layer_guid_t const &l,
                                   IncomingTensorRole desired_role) {
  PCGOperatorAttrs attrs = get_parallel_layer_attrs(pcg, l).op_attrs;

  std::unordered_map<TensorSlotName, parallel_tensor_guid_t> incoming_tensors =
      get_incoming_tensors(pcg, l);

  std::unordered_map<TensorSlotName, IncomingTensorRole> incoming_slot_roles =
      get_incoming_tensor_roles(attrs);

  ASSERT(incoming_tensors.size() == incoming_slot_roles.size());

  std::unordered_set<TensorSlotName> slots_with_desired_role =
      keys(filter_values(incoming_slot_roles, [&](IncomingTensorRole role) {
        return role == desired_role;
      }));

  return restrict_keys(incoming_tensors, slots_with_desired_role);
}

std::unordered_map<TensorSlotName, parallel_tensor_guid_t>
    get_incoming_inputs(ParallelComputationGraph const &pcg,
                        parallel_layer_guid_t const &l) {
  return get_incoming_tensors_with_role(pcg, l, IncomingTensorRole::INPUT);
}

std::unordered_map<TensorSlotName, parallel_tensor_guid_t>
    get_incoming_weights(ParallelComputationGraph const &pcg,
                         parallel_layer_guid_t const &l) {
  return get_incoming_tensors_with_role(pcg, l, IncomingTensorRole::WEIGHT);
}

std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>
    get_incoming_input_degrees(ParallelComputationGraph const &pcg,
                               parallel_layer_guid_t const &l) {

  return map_values(get_incoming_inputs(pcg, l), [&](parallel_tensor_guid_t t) {
    return get_parallel_degrees(get_parallel_tensor_shape(pcg, t));
  });
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

std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>
    get_parallel_layer_attrs_mapping(ParallelComputationGraph const &pcg) {
  std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>
      layer_attrs_mapping;
  for (parallel_layer_guid_t const &layer_guid : get_parallel_layers(pcg)) {
    layer_attrs_mapping.insert(
        {layer_guid, get_parallel_layer_attrs(pcg, layer_guid)});
  }
  return layer_attrs_mapping;
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
      LabelledKwargDataflowGraph<ParallelLayerAttrs,
                                 ParallelTensorAttrs,
                                 TensorSlotName>::
          create_copy_of<
              UnorderedSetLabelledOpenKwargDataflowGraph<ParallelLayerAttrs,
                                                         ParallelTensorAttrs,
                                                         int,
                                                         TensorSlotName>>(
              rewrite_labelled_kwarg_dataflow_graph_node_labels(
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
  return find_isomorphism_between_kwarg_dataflow_graphs(
             without_layer_names(lhs).raw_graph,
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

  return labelled_open_kwarg_dataflow_graph_view_as_dot(
      view_as_labelled_open_kwarg_dataflow_graph<ParallelLayerAttrs,
                                                 ParallelTensorAttrs,
                                                 int,
                                                 TensorSlotName>(cg.raw_graph),
      get_node_label,
      get_input_label);
}

void debug_print_dot(ParallelComputationGraph const &cg) {
  std::cout << as_dot(cg) << std::endl;
}

} // namespace FlexFlow
