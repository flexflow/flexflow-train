#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/operator_task_space.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include <unordered_set>

namespace FlexFlow {

OpCostEstimateKey get_mapped_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg,
    parallel_layer_guid_t const &layer,
    MachineView const &machine_view) {
  return map_unmapped_op_cost_estimate_key(
      get_unmapped_op_cost_estimate_key_for_layer(pcg, layer), machine_view);
}

} // namespace FlexFlow
