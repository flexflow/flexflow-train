#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"

namespace FlexFlow {

MachineMapping combine_disjoint_mappings(MachineMapping const &s1,
                                         MachineMapping const &s2) {
  return MachineMapping{merge_maps(s1.machine_views, s2.machine_views)};
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

MachineMapping get_machine_mapping_from_machine_mapping_result(
    PCGBinarySPDecomposition const &sp_decomposition,
    MachineMappingResult const &mm_result) {

  BinarySPDecompositionTree sp_tree =
      binary_sp_tree_from_pcg_sp_tree(sp_decomposition);

  auto get_layer_from_path =
      [&](BinaryTreePath const &path) -> parallel_layer_guid_t {
    std::optional<BinarySPDecompositionTree> subtree_optional =
        binary_sp_decomposition_tree_get_subtree_at_path(sp_tree, path);
    if (!subtree_optional.has_value()) {
      throw std::runtime_error(fmt::format("Invalid tree path {}", path));
    }
    BinarySPDecompositionTree subtree = subtree_optional.value();
    if (!subtree.is_node()) {
      throw std::runtime_error(fmt::format(
          "Invalid tree path to a leaf: found {} instead", subtree));
    }
    return parallel_layer_guid_t{
        subtree.require_node(),
    };
  };

  std::unordered_map<parallel_layer_guid_t, MachineView> mm;

  if (mm_result.raw_result) {
    FeasibleMachineMappingResult const &feasible_mm_result =
        mm_result.raw_result.value();
    mm = map_keys(feasible_mm_result.machine_mapping.raw_mapping,
                  get_layer_from_path);
  }

  return MachineMapping{mm};
}

} // namespace FlexFlow
