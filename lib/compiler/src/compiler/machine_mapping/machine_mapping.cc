#include "compiler/machine_mapping/machine_mapping.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_maps.h"

namespace FlexFlow {

MachineMapping combine_disjoint_mappings(MachineMapping const &m1,
                                         MachineMapping const &m2) {
  return MachineMapping{
      merge_disjoint_maps(m1.machine_views, m2.machine_views)};
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

} // namespace FlexFlow
