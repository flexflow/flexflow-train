#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/accessors_are_approximately_equal.h"
#include "kernels/emulated_parallel_tensor/emulated_parallel_tensor.h"
#include "kernels/emulated_parallel_tensor/parallel_tensor_reduction.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/get_shard_signature_instance.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/shape_inference.h"
#include "utils/containers/all_of.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/require_same.h"
#include "utils/containers/zip_values_strict_with.h"
#include <random>

namespace FlexFlow {

bool shard_signature_instance_is_valid(
    ComputationGraphOpAttrs const &attrs,
    std::map<TensorSlotName, ParallelTensorShape> const &input_shapes,
    std::function<std::map<TensorSlotName, GenericTensorAccessorR>(
        std::map<TensorSlotName, GenericTensorAccessorR> const &)> const
        &run_op,
    int seed) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  std::map<TensorSlotName, ParallelTensorShape> weight_shapes =
      get_weight_shapes(pcg_op_attrs_from_compgraph_op_attrs(attrs),
                        input_shapes);

  std::map<TensorSlotName, ParallelTensorShape> incoming_shapes =
      binary_merge_disjoint_maps(input_shapes, weight_shapes);

  std::mt19937 gen(seed);

  std::map<TensorSlotName, int> incoming_seeds =
      generate_map(keys(incoming_shapes),
                   [&](TensorSlotName const &) -> int { return gen(); });

  std::map<TensorSlotName, ParallelTensorDimDegrees> input_degrees = map_values(
      input_shapes,
      [&](ParallelTensorShape const &shape) -> ParallelTensorDimDegrees {
        return get_parallel_degrees(shape);
      });

  ShardSignatureInstance shard_signature_instance =
      get_shard_signature_instance(attrs, input_degrees);

  {
    // TODO(@lockshaw): add additional checks for output shapes
    std::set<TensorSlotName> incoming_slot_names = keys(incoming_shapes);

    std::map<TensorSlotName, ParallelTensorDimDegrees> incoming_degrees =
        generate_map(incoming_slot_names,
                     [&](TensorSlotName slot_name) -> ParallelTensorDimDegrees {
                       return get_parallel_degrees(
                           incoming_shapes.at(slot_name));
                     });

    std::map<TensorSlotName, ParallelTensorDimDegrees>
        incoming_degrees_according_to_signature = generate_map(
            incoming_slot_names,
            [&](TensorSlotName slot_name) -> ParallelTensorDimDegrees {
              return parallel_tensor_space_for_shard_signature_instance_and_slot(
                  shard_signature_instance, slot_name);
            });

    ASSERT(incoming_degrees == incoming_degrees_according_to_signature);
  }

  std::map<TensorSlotName, EmulatedParallelTensor> incoming =
      zip_values_strict_with(incoming_shapes,
                             incoming_seeds,
                             [&](ParallelTensorShape const &shape,
                                 int seed) -> EmulatedParallelTensor {
                               return random_emulated_parallel_tensor_of_shape(
                                   shape, cpu_allocator, seed);
                             });

  auto unparallelize =
      [&](std::map<TensorSlotName, EmulatedParallelTensor> const
              &parallel_tensors)
      -> std::map<TensorSlotName, GenericTensorAccessorR> {
    return map_values(
        parallel_tensors,
        [&](EmulatedParallelTensor const &ptensor) -> GenericTensorAccessorR {
          return unparallelize_parallel_tensor(ptensor, cpu_allocator);
        });
  };

  std::map<TensorSlotName, GenericTensorAccessorR> op_then_unpar =
      unparallelize(parallelize_tensor_operation(
          incoming, shard_signature_instance.get_shard_bindings(), run_op));

  std::map<TensorSlotName, GenericTensorAccessorR> unpar_then_op =
      run_op(unparallelize(incoming));

  std::set<TensorSlotName> output_slots =
      require_same(keys(op_then_unpar), keys(unpar_then_op));

  return all_of(output_slots, [&](TensorSlotName output_slot_name) -> bool {
    GenericTensorAccessorR slot_op_then_unpar =
        op_then_unpar.at(output_slot_name);
    GenericTensorAccessorR slot_unpar_then_op =
        unpar_then_op.at(output_slot_name);

    return accessors_are_approximately_equal(slot_op_then_unpar,
                                             slot_unpar_then_op);
  });
}

} // namespace FlexFlow
