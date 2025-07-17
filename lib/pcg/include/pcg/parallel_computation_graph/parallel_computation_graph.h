#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_added_result.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include <unordered_set>

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph();

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(ParallelComputationGraph const &);
std::unordered_set<parallel_tensor_guid_t>
    get_parallel_tensors(ParallelComputationGraph const &);

ParallelLayerAddedResult add_parallel_layer(
    ParallelComputationGraph &pcg,
    ParallelLayerAttrs const &layer_attrs,
    std::vector<parallel_tensor_guid_t> const &inputs,
    std::vector<parallel_tensor_guid_t> const &weights,
    std::optional<std::vector<CreateGrad>> const &outputs = std::nullopt);

ParallelLayerAddedResult pcg_add_input_layer(ParallelComputationGraph &pcg,
                                             TensorShape const &tensor_shape);

std::unordered_set<ParallelComputationGraphEdge>
    get_pcg_edges_from_layer_to_layer(ParallelComputationGraph const &,
                                      parallel_layer_guid_t const &,
                                      parallel_layer_guid_t const &);

std::unordered_set<ParallelComputationGraphEdge>
    get_edges(ParallelComputationGraph const &);

std::unordered_set<ParallelComputationGraphEdge>
    get_outgoing_edges(ParallelComputationGraph const &,
                       parallel_layer_guid_t const &);

std::unordered_set<ParallelComputationGraphEdge>
    get_incoming_edges(ParallelComputationGraph const &,
                       parallel_layer_guid_t const &);

std::unordered_set<parallel_layer_guid_t>
    get_initial_layers(ParallelComputationGraph const &);

std::vector<parallel_tensor_guid_t>
    get_incoming_tensors(ParallelComputationGraph const &,
                         parallel_layer_guid_t const &);
std::vector<parallel_tensor_guid_t>
    get_layer_outputs(ParallelComputationGraph const &,
                      parallel_layer_guid_t const &);

std::vector<parallel_tensor_guid_t>
    get_incoming_inputs(ParallelComputationGraph const &,
                        parallel_layer_guid_t const &);
std::vector<parallel_tensor_guid_t>
    get_incoming_weights(ParallelComputationGraph const &,
                         parallel_layer_guid_t const &);

std::unordered_set<parallel_layer_guid_t>
    get_successors(ParallelComputationGraph const &,
                   parallel_layer_guid_t const &);

std::unordered_set<parallel_layer_guid_t>
    get_subgraph_successors(ParallelComputationGraph const &,
                            std::unordered_set<parallel_layer_guid_t> const &);

parallel_layer_guid_t get_source_layer(ParallelComputationGraph const &g,
                                       parallel_tensor_guid_t const &t);

ParallelLayerAttrs get_parallel_layer_attrs(ParallelComputationGraph const &,
                                            parallel_layer_guid_t const &);
PCGOperatorAttrs pcg_get_op_attrs(ParallelComputationGraph const &,
                                  parallel_layer_guid_t const &);
ParallelTensorAttrs get_parallel_tensor_attrs(ParallelComputationGraph const &,
                                              parallel_tensor_guid_t const &);
ParallelTensorShape get_parallel_tensor_shape(ParallelComputationGraph const &,
                                              parallel_tensor_guid_t const &);

std::vector<parallel_layer_guid_t>
    topological_ordering(ParallelComputationGraph const &);

parallel_layer_guid_t
    get_parallel_layer_by_name(ParallelComputationGraph const &pcg,
                               std::string const &name);

ParallelComputationGraph without_layer_names(ParallelComputationGraph const &);

bool pcgs_are_isomorphic(ParallelComputationGraph const &,
                         ParallelComputationGraph const &);

std::string as_dot(ParallelComputationGraph const &);
void debug_print_dot(ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
