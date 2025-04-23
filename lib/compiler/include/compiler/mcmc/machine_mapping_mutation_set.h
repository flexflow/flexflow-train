#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H

#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/search_result.dtg.h"

namespace FlexFlow {
std::vector<MachineMapping>
    get_possible_mutations(SearchResult mapped_pcg,
                           MachineSpecification const &resource);
MachineMapping permute_layers(std::vector<parallel_layer_guid_t> layers,
                              MachineMapping mapping);
MachineMapping copy_layer(parallel_layer_guid_t source,
                          parallel_layer_guid_t destination,
                          MachineMapping mapping);
MachineView change_stride(nonnegative_int stride,
                          parallel_layer_guid_t layer,
                          MachineView machine_view,
                          nonnegative_int dim);
MachineView change_node_idx(nonnegative_int node_ix,
                            parallel_layer_guid_t layer,
                            MachineView machine_view);
MachineView change_device_idx(nonnegative_int device_idx,
                              parallel_layer_guid_t layer,
                              MachineView machine_view);
MachineView change_projection(MachineSpecificationDimension projection,
                              parallel_layer_guid_t layer,
                              MachineView machine_view,
                              nonnegative_int dim);
} // namespace FlexFlow

#endif
