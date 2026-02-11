#include "local-execution/task_execution.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/optimizer_attrs.h"
#include "pcg/optimizer_slot_name.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include "utils/overload.h"
#include <optional>

namespace FlexFlow {

TaskTensorParameter make_task_tensor_parameter_dynamic(
    DynamicTensorSlot slot,
    std::optional<OptimizerAttrs> const &optimizer_attrs) {
  return assert_unwrap(slot.slot_tensor_role)
      .visit<TaskTensorParameter>(overload{
          [&](FwbTensorType const &fwb_tensor) {
            switch (fwb_tensor) {
              case FwbTensorType::FORWARD:
                return make_task_tensor_parameter_fwd(slot.slot_name);
              case FwbTensorType::GRADIENT:
                return make_task_tensor_parameter_grad(slot.slot_name);
              default:
                PANIC("Unhandled FwbTensorType", fmt::to_string(fwb_tensor));
            }
          },
          [&](DynamicOptimizerTensorRole const &optimizer_tensor) {
            return make_task_tensor_parameter_opt(
                slot.slot_name, optimizer_tensor.optimizer_slot_name);
          },
          [&](DynamicLossTensorRole const &loss_tensor) {
            return make_task_tensor_parameter_loss();
          },
      });
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
  std::unordered_map<TaskTensorParameter, DynamicTensorAccessor>
      tensor_slots_backing;
  for (auto const &[slot, input] : invocation.inputs) {
    TaskTensorParameter param =
        make_task_tensor_parameter_dynamic(slot, optimizer_attrs);
    DynamicTensorAccessor accessor = assert_unwrap(input.accessor);
    bool ok = tensor_slots_backing.insert(std::pair{param, accessor}).second;
    ASSERT(ok);
  }
  for (auto const &[slot, output] : invocation.outputs) {
    TaskTensorParameter param =
        make_task_tensor_parameter_dynamic(slot, optimizer_attrs);
    DynamicTensorAccessor accessor = assert_unwrap(output.accessor);
    bool ok = tensor_slots_backing.insert(std::pair{param, accessor}).second;
    ASSERT(ok);
  }

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      /*allocator=*/allocator,
      /*tensor_slots_backing=*/tensor_slots_backing,
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*op_attrs=*/invocation.node_attrs.op_attrs,
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
  std::optional<milliseconds_t> result;
  switch (task_type) {
    case DynamicTaskType::FWD: {
      ComputationGraphOpAttrs op_attrs =
          assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
              assert_unwrap(invocation.node_attrs.op_attrs)));
      result = call_fwd_task_impl(op_attrs, arg_accessor);
    } break;
    case DynamicTaskType::BWD: {
      ComputationGraphOpAttrs op_attrs =
          assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
              assert_unwrap(invocation.node_attrs.op_attrs)));
      result = call_bwd_task_impl(op_attrs, arg_accessor);
    } break;
    case DynamicTaskType::UPD:
      call_update_task_impl(assert_unwrap(optimizer_attrs), arg_accessor);
      break;
    case DynamicTaskType::LOSS:
      call_loss_task_impl(arg_accessor);
      break;
    default:
      PANIC("Unhandled DynamicTaskType", task_type);
  }
  return result;
}

} // namespace FlexFlow
