#include "task-spec/runtime_task_invocation/runtime_arg_spec.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "task-spec/runtime_task_invocation/runtime_arg_ref.h"
#include "utils/overload.h"

namespace FlexFlow {

std::type_index get_type_index(RuntimeArgSpec const &task_arg_spec) {
  return task_arg_spec.visit<std::type_index>(
      overload{[](auto const &e) { return e.get_type_index(); }});
}

// TODO(@lockshaw)(#pr): 
// RuntimeArgSpec lower_op_arg_spec_to_runtime_arg_spec(
//     OpArgSpec const &op_arg_spec,
//     symbolic_layer_guid_t symbolic_layer_guid,
//     SymbolicLayerTensorShapeSignature const &op_shape_signature) {
//   return op_arg_spec.visit<RuntimeArgSpec>(overload{
//       [](ConcreteArgSpec const &concrete_arg_spec) -> RuntimeArgSpec {
//         return RuntimeArgSpec{concrete_arg_spec};
//       },
//       [](RuntimeArgRefSpec const &runtime_arg_ref_spec) -> RuntimeArgSpec {
//         return RuntimeArgSpec{runtime_arg_ref_spec};
//       },
//       [&](OpArgRefSpec const &op_arg_ref_spec) -> RuntimeArgSpec {
//         return 
//             lower_op_arg_ref_spec_to_runtime_arg_spec(op_arg_ref_spec,
//                                        symbolic_layer_guid,
//                                        op_shape_signature);
//       },
//   });
// }
//
// RuntimeArgSpec lower_op_arg_ref_spec_to_runtime_arg_spec(
//     OpArgRefSpec const &op_arg_ref_spec,
//     symbolic_layer_guid_t symbolic_layer_guid,
//     SymbolicLayerTensorShapeSignature const &op_signature) {
//
//   OpArgRefType op_arg_ref_type = op_arg_ref_spec.get_ref_type();
//   return op_arg_ref_type.visit<RuntimeArgSpec>(overload{
//       [&](PerDeviceOpStateRefType const &) -> RuntimeArgSpec {
//         return RuntimeArgSpec{
//           RuntimeArgRefSpec::create(per_device_op_state_for_layer(symbolic_layer_guid)),
//         };
//       },
//       [&](ParallelTensorShapeRefType const &ref_type) -> RuntimeArgSpec {
//         TensorShape tensor_shape = tensor_shape_for_role_and_index(
//             /*signature=*/op_signature,
//             /*tensor_role=*/ref_type.tensor_role,
//             /*index=*/ref_type.idx);
//         ParallelTensorShape shape = lift_to_parallel(tensor_shape);
//         return RuntimeArgSpec{
//           ConcreteArgSpec::create(shape),
//         };
//       },
//   });
// }

} // namespace FlexFlow
