#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

std::unordered_set<parallel_layer_guid_t>
    mpcg_get_parallel_layers(MappedParallelComputationGraph const &);

MappedOperatorTaskGroup
    mpcg_get_mapping_for_layer(MappedParallelComputationGraph const &,
                               parallel_layer_guid_t);

ParallelComputationGraph pcg_from_mpcg(MappedParallelComputationGraph const &);

parallel_layer_guid_t mpcg_get_source_layer(MappedParallelComputationGraph const &,
                                            parallel_tensor_guid_t const &);

ParallelTensorAttrs mpcg_get_parallel_tensor_attrs(MappedParallelComputationGraph const &,
                                                   parallel_tensor_guid_t const &);

std::unordered_map<TensorSlotName, ParallelComputationGraphEdge>
  mpcg_get_incoming_edges(MappedParallelComputationGraph const &,
                          parallel_layer_guid_t const &);

std::unordered_set<ParallelComputationGraphEdge>
  mpcg_get_outgoing_edges(MappedParallelComputationGraph const &,
                          parallel_layer_guid_t const &);

ManyToOne<TensorSlotName, parallel_tensor_guid_t>
    mpcg_get_incoming_tensors(MappedParallelComputationGraph const &,
                              parallel_layer_guid_t const &);

bidict<TensorSlotName, parallel_tensor_guid_t>
    mpcg_get_outgoing_tensors(MappedParallelComputationGraph const &,
                         parallel_layer_guid_t const &);

std::unordered_set<ParallelComputationGraphEdge>
    mpcg_get_edges(MappedParallelComputationGraph const &);

MappedParallelComputationGraph mapped_pcg_from_pcg_and_mapped_op_task_groups(
    ParallelComputationGraph const &pcg,
    std::unordered_map<parallel_layer_guid_t, MappedOperatorTaskGroup> const
        &mapped_op_task_groups);

MappedParallelComputationGraph
    mapped_pcg_without_layer_names(MappedParallelComputationGraph const &);

std::string format_as(MappedParallelComputationGraph const &);
std::ostream &operator<<(std::ostream &,
                         MappedParallelComputationGraph const &);

bool mapped_pcgs_are_isomorphic(MappedParallelComputationGraph const &,
                                MappedParallelComputationGraph const &);

std::string mapped_pcg_as_dot(MappedParallelComputationGraph const &);
void debug_print_mapped_pcg_as_dot(MappedParallelComputationGraph const &);

} // namespace FlexFlow

#endif
