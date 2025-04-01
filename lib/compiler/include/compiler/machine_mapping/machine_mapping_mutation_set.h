#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H

#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/search_result.dtg.h"

namespace FlexFlow {
std::optional<MachineMapping>
    get_naive_mapping(ParallelComputationGraph &pcg,
                      MachineSpecification const &resources);
std::vector<MachineMapping>
    get_possible_mutations(SearchResult mapped_pcg,
                           MachineSpecification const &resource);
std::optional<MachineMapping>
    get_random_mutation(SearchResult mapped_pcg,
                        MachineSpecification const &resource,
                        DeviceType const &device_type = DeviceType::GPU);
MachineView increment_stride(MachineView machine_view, nonnegative_int dim);
MachineView decrement_all_strides(MachineView machine_view);
MachineView change_stride(nonnegative_int stride,
                          MachineView machine_view,
                          nonnegative_int dim);
MachineView change_node_idx(nonnegative_int node_ix, MachineView machine_view);
MachineView change_device_idx(nonnegative_int device_idx,
                              MachineView machine_view);
MachineView switch_projection(MachineView machine_view, nonnegative_int dim);
} // namespace FlexFlow

#endif
