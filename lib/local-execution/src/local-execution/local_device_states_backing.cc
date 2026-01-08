#include "local-execution/local_device_states_backing.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/local_tensor_backing.h"
#include "task-spec/task_signature_impl.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/overload.h"

namespace FlexFlow {

// LocalDeviceStatesBacking
// make_local_device_states_backing_for_computation_graph(
//     LocalTaskRegistry const &task_registry,
//     std::unordered_map<layer_guid_t,
//     SymbolicLayerTrainingTensorGroupSignatureWithShapes> const &layers,
//     std::unordered_map<layer_guid_t, ComputationGraphOpAttrs> const
//     &op_attrs, RuntimeArgConfig const &runtime_arg_config, LocalTensorBacking
//     const &local_tensor_backing, Allocator &allocator) {
//
//   std::unordered_map<layer_guid_t,
//   std::optional<DeviceSpecificPerDeviceOpState>>
//       per_device_op_states = generate_map(
//           keys(layers),
//           [&](layer_guid_t const &layer_guid) ->
//           std::optional<DeviceSpecificPerDeviceOpState> {
//             return create_per_device_op_state(
//                 task_registry,
//                 local_tensor_backing,
//                 runtime_arg_config,
//                 allocator,
//                 op_attrs,
//                 layers.at(layer_guid));
//           });
//
//   return LocalDeviceStatesBacking{
//     per_device_op_states,
//   };
// }

// std::optional<DeviceSpecificPerDeviceOpState>
// get_per_device_op_state_if_exists(
//     LocalArgsBacking const &local_args_backing,
//     layer_guid_t const &layer_guid) {
//
//   return local_args_backing.per_device_op_states.at(layer_guid);
// }

} // namespace FlexFlow
