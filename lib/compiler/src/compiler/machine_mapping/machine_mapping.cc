#include "compiler/machine_mapping/machine_mapping.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/operator_task_space.dtg.h"
#include "pcg/operator_task_space.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/get_one_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_maps.h"

namespace FlexFlow {

MachineMapping combine_disjoint_mappings(MachineMapping const &m1,
                                         MachineMapping const &m2) {
  return MachineMapping{merge_maps(m1.machine_views, m2.machine_views)};
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

} // namespace FlexFlow
