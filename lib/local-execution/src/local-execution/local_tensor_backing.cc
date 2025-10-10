#include "local-execution/local_tensor_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/fwb_tensor_slot_id_t.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/is_submapeq_of.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/overload.h"

namespace FlexFlow {

// LocalTensorBacking construct_local_tensor_backing(
//     std::unordered_map<symbolic_training_tensor_guid_t, TensorShape> const
//         &training_tensor_shapes,
//     std::unordered_map<symbolic_training_tensor_guid_t, GenericTensorAccessorW> const
//         &preallocated,
//     Allocator &allocator) {
//
//   ASSERT(is_subseteq_of(keys(preallocated), keys(training_tensor_shapes)));
//
//   std::unordered_set<symbolic_training_tensor_guid_t> to_allocate =
//       set_minus(keys(training_tensor_shapes), keys(preallocated));
//
//   std::unordered_map<symbolic_training_tensor_guid_t, GenericTensorAccessorW> allocated =
//       generate_map(to_allocate, [&](training_tensor_guid_t t) {
//         TensorShape shape = training_tensor_shapes.at(t);
//         return allocator.allocate_tensor(shape);
//       });
//
//   std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW>
//       backing_for_training_tensor_map =
//           merge_disjoint_maps(allocated, preallocated);
//
//   ASSERT(is_submapeq_of(preallocated, backing_for_training_tensor_map));
//
//   ASSERT(keys(backing_for_training_tensor_map) == keys(training_tensor_shapes),
//          backing_for_training_tensor_map.size(),
//          training_tensor_shapes.size(),
//          keys(preallocated));
//
//   return LocalTensorBacking{
//       backing_for_training_tensor_map,
//   };
// }

AtomicTaskInvocation 
  lower_local_runtime_task_invocation_to_atomic_task_invocation(
    LocalTensorBacking const &,
    RuntimeTaskInvocation const &,
    RuntimeArgConfig const &) {
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
