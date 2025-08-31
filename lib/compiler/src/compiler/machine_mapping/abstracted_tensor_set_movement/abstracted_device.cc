#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_device.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.dtg.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/machine_view.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

MachineSpaceCoordinate
  concretize_abstracted_device(
    AbstractedDevice const &abstracted_device,
    std::unordered_map<BinaryTreePath, OperatorTaskSpace> const &task_spaces,
    ParallelLayerGuidObliviousMachineMapping const &machine_mapping) {

  return get_machine_space_coordinate(
    /*operator_task_space=*/task_spaces.at(abstracted_device.operator_tree_path),
    /*machine_view=*/machine_mapping.raw_mapping.at(abstracted_device.operator_tree_path),
    /*task_space_coordinate=*/abstracted_device.task_space_coordinate);
}

} // namespace FlexFlow
