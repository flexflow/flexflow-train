#include "local-execution/task_execution.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/optimizer_attrs.h"
#include "pcg/optimizer_slot_name.dtg.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

std::unordered_set<TaskTensorParameter> make_task_tensor_parameter_dynamic(
    TensorSlotName slot_name,
    DynamicTaskType task_type,
    std::optional<OptimizerAttrs> const &optimizer_attrs) {
  switch (task_type) {
    case DynamicTaskType::FWD:
      return std::unordered_set{make_task_tensor_parameter_fwd(slot_name)};
    case DynamicTaskType::BWD:
      return std::unordered_set{make_task_tensor_parameter_grad(slot_name)};
    case DynamicTaskType::UPD:
      return transform(
          get_slot_names_for_optimizer(assert_unwrap(optimizer_attrs)),
          [&](OptimizerSlotName optimizer_slot) {
            return make_task_tensor_parameter_opt(slot_name, optimizer_slot);
          });
    default:
      PANIC("Unhandled DynamicTaskType", fmt::to_string(task_type));
  }
}

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    device_id_t device_idx) {
  PCGOperatorAttrs op_attrs = assert_unwrap(invocation.node_attrs.op_attrs);

  std::unordered_map<TaskTensorParameter, DynamicTensorAccessor>
      tensor_slots_backing;
  for (auto const &[slot, input] : invocation.inputs) {
    std::unordered_set<TaskTensorParameter> params =
        make_task_tensor_parameter_dynamic(
            slot.slot_name,
            assert_unwrap(invocation.node_attrs.task_type),
            optimizer_attrs);
    DynamicTensorAccessor accessor = assert_unwrap(input.accessor);
    for (auto const &param : params) {
      tensor_slots_backing.insert(std::pair{param, accessor});
    }
  }
  for (auto const &[slot, output] : invocation.outputs) {
    std::unordered_set<TaskTensorParameter> params =
        make_task_tensor_parameter_dynamic(
            slot.slot_name,
            assert_unwrap(invocation.node_attrs.task_type),
            optimizer_attrs);
    DynamicTensorAccessor accessor = assert_unwrap(output.accessor);
    for (auto const &param : params) {
      tensor_slots_backing.insert(std::pair{param, accessor});
    }
  }

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      /*allocator=*/allocator,
      /*tensor_slots_backing=*/tensor_slots_backing,
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*op_attrs=*/op_attrs,
      /*loss_attrs=*/loss_attrs,
      /*per_device_op_state=*/per_device_op_state,
      /*iteration_config=*/iteration_config,
      /*optimizer_attrs=*/optimizer_attrs,
      /*device_idx=*/device_idx);
}

std::optional<milliseconds_t> execute_dynamic_node_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    device_id_t device_idx) {
  TaskArgumentAccessor arg_accessor =
      make_task_argument_accessor_for_invocation(
          /*invocation=*/invocation,
          /*allocator=*/allocator,
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/ff_handle,
          /*loss_attrs=*/loss_attrs,
          /*per_device_op_state=*/per_device_op_state,
          /*iteration_config=*/iteration_config,
          /*optimizer_attrs=*/optimizer_attrs,
          /*device_idx=*/device_idx);

  DynamicTaskType task_type = assert_unwrap(invocation.node_attrs.task_type);
  ComputationGraphOpAttrs op_attrs =
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
          assert_unwrap(invocation.node_attrs.op_attrs)));
  std::optional<milliseconds_t> result;
  switch (task_type) {
    case DynamicTaskType::FWD:
      result = call_fwd_task_impl(op_attrs, arg_accessor);
      break;
    case DynamicTaskType::BWD:
      result = call_bwd_task_impl(op_attrs, arg_accessor);
      break;
    case DynamicTaskType::UPD:
      NOT_IMPLEMENTED();
      break;
    default:
      PANIC("Unhandled DynamicTaskType", fmt::to_string(task_type));
  }
  return result;
}

} // namespace FlexFlow
