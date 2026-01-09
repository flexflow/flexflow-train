#include "local-execution/computation_graph_instance/initialized_computation_graph_instance.h"
#include "utils/exception.h"

namespace FlexFlow {

InitializedComputationGraphInstance initialize_computation_graph_instance(
    ComputationGraphInstance const &instance,
    bidict<tensor_guid_t,
           std::variant<GenericTensorAccessorW, GenericTensorAccessorR>> const
        &input_tensors,
    Allocator &allocator) {
  NOT_IMPLEMENTED();
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
