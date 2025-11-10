#include "local-execution/computation_graph_instance/initialized_computation_graph_instance.h"
// #include "local-execution/execute_pass.h"

namespace FlexFlow {

std::unordered_map<
  layer_guid_t,
  std::optional<milliseconds_t>> 
    perform_forward_pass_for_computation_graph_instance(
      InitializedComputationGraphInstance const &instance) {

  // TODO(@lockshaw)(#pr): 
  NOT_IMPLEMENTED();
  // return execute_forward_pass(
  //   instance.get_symbolic_training_graph_for_cg(),
  //   instance.get_tensor_backing(),
  //   instance.get_atomic_tensor_backing(),
  //   instance.get_allocator(),
  //   instance.get_task_registry(),
  //   instance.get_runtime_arg_config());
}

std::unordered_map<
  layer_guid_t,
  std::optional<milliseconds_t>> 
    perform_backward_pass_for_computation_graph_instance(
      InitializedComputationGraphInstance const &instance) {

  // TODO(@lockshaw)(#pr): 
  // return execute_backward_pass(
  //   instance.get_symbolic_training_graph_for_cg(),
  //   instance.get_tensor_backing(),
  //   instance.get_atomic_tensor_backing(),
  //   instance.get_allocator(),
  //   instance.get_task_registry(),
  //   instance.get_runtime_arg_config());
}

} // namespace FlexFlow
