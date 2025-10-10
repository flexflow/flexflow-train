#include "local-execution/local_training_backing.h"
#include "local-execution/local_args_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/loss_functions.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/optimizer.h"
#include "task-spec/task_invocation.h"
#include "task-spec/task_signature_impl.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/values.h"
#include "utils/exception.h"

namespace FlexFlow {

LocalTrainingBacking make_local_training_backing_for_computation_graph(
    Allocator &allocator,
    std::unordered_map<symbolic_training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated,
    TrainingComputationGraph const &training_computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs) {

  ASSERT(is_subseteq_of(
      keys(preallocated),
      keys(get_all_training_tensor_shapes(training_computation_graph))));

  LocalTaskRegistry local_task_registry =
      construct_local_task_registry_for_layers(get_layer_attrs_mapping(
          training_computation_graph.computation_graph));

  LocalTensorBacking local_tensor_backing = construct_local_tensor_backing(
      get_all_training_tensor_shapes(training_computation_graph),
      preallocated,
      allocator);

  LocalArgsBacking local_args_backing =
      make_local_args_backing_for_computation_graph(local_task_registry,
                                                    training_computation_graph,
                                                    runtime_arg_config,
                                                    local_tensor_backing,
                                                    allocator);

  return LocalTrainingBacking{
      /*computation_graph=*/training_computation_graph,
      /*local_task_registry=*/local_task_registry,
      /*local_tensor_backing=*/local_tensor_backing,
      /*local_args_backing=*/local_args_backing,
  };
}

std::optional<DeviceSpecificPerDeviceOpState>
    create_per_device_op_state(LocalTaskRegistry const &local_task_registry,
                               LocalTensorBacking const &tensor_backing,
                               RuntimeArgConfig const &runtime_arg_config,
                               Allocator &allocator,
                               layer_guid_t layer_id,
                               ComputationGraphOpAttrs const &op_attrs,
                               SymbolicLayerTrainingTensorGroupSignatureWithShapes const &layer_signature) {

  std::optional maybe_registered_task = try_get_registered_task(
      local_task_registry, layer_id, OpTaskType::INIT);

  ASSERT(maybe_registered_task.has_value());
  registered_task_t registered_task = maybe_registered_task.value();

  OpTaskInvocation op_init_task_invocation =
    get_init_op_task_invocation(op_attrs);

  TaskInvocation invocation = lower_op_task_invocation_to_task_invocation(
      /*op_task_invocation=*/op_init_task_invocation,
      /*layer_signature=*/layer_signature,
      /*device_specific_device_states=*/std::nullopt);

  TaskArgumentAccessor accessor = get_task_arg_accessor_for_invocation(
      tensor_backing, runtime_arg_config, invocation, allocator);
  
  return call_init_task_impl(
    local_task_registry,
    registered_task,
    accessor); 
}

void execute_update(LocalTrainingBacking const &local_training_backing,
                    layer_guid_t const &layer_guid,
                    OptimizerAttrs const &optimizer_attrs,
                    Allocator &allocator) {
  TrainingLayerPlusContext training_layer = get_training_layer_plus_context(
      local_training_backing.training_computation_graph, layer_guid);

  if (training_layer.layer_attrs.op_attrs.has<WeightAttrs>()) {
    TrainingTensorGroupWithAttrs weight_tensor_group =
        get_only(training_layer.output_tensor_groups);

    TaskInvocation invocation =
        get_update_invocation(optimizer_attrs,
                              weight_tensor_group.forward_tensor,
                              weight_tensor_group.gradient_tensor,
                              weight_tensor_group.optimizer_tensors);

    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    TaskArgumentAccessor accessor = get_task_arg_accessor(
        local_training_backing.local_tensor_backing,
        local_training_backing.local_args_backing.runtime_arg_config,
        invocation,
        allocator);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    update_impl_fn.get<GenericTaskImplFunction>().function_ptr(accessor);
  }
}

} // namespace FlexFlow
