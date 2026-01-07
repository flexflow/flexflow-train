#include "local-execution/computation_graph_instance/initialized_computation_graph_instance.h"

namespace FlexFlow {

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
