#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/operator_task_space.dtg.h"
#include "pcg/operator_task_space.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/transform.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_subtree_at_path.h"

namespace FlexFlow {

MachineMapping combine_disjoint_mappings(MachineMapping const &m1,
                                         MachineMapping const &m2) {
  return MachineMapping{
      merge_disjoint_maps(m1.machine_views, m2.machine_views)};
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

parallel_layer_guid_t
    get_layer_from_path(PCGBinarySPDecomposition const &sp_decomposition,
                        BinaryTreePath const &path) {
  std::optional<PCGBinarySPDecomposition> subtree_optional =
      get_subtree_at_path(
          sp_decomposition, generic_impl_for_pcg_sp_tree(), path);

  if (!subtree_optional.has_value()) {
    throw std::runtime_error(fmt::format("Invalid tree path {}", path));
  }

  PCGBinarySPDecomposition subtree = subtree_optional.value();
  if (!subtree.is_leaf()) {
    throw std::runtime_error(
        fmt::format("Invalid tree path to a leaf: found {} instead", subtree));
  }
  return subtree.require_leaf();
}

std::optional<MachineMapping> get_machine_mapping_from_machine_mapping_result(
    PCGBinarySPDecomposition const &sp_decomposition,
    MachineMappingResult const &mm_result) {

  return transform(
      mm_result.raw_result,
      [&](FeasibleMachineMappingResult const &feasible_mm_result) {
        return MachineMapping{
            map_keys(feasible_mm_result.machine_mapping.raw_mapping,
                     [&](BinaryTreePath const &path) {
                       return get_layer_from_path(sp_decomposition, path);
                     }),
        };
      });
}

} // namespace FlexFlow
