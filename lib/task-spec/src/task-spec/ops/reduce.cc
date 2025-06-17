#include "task-spec/ops/reduce.h"
#include "kernels/reduce_kernels.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Reduce;

enum Slots {
  INPUT,
  OUTPUT,
  ATTRS,
  PROFILING,
  REDUCE,
  PER_DEVICE_STATE,
  HANDLE,
  KERNEL_DEVICE_TYPE,
};

OpTaskInvocation init(ReduceAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return OpTaskInvocation{
    task_id_t::REDUCE_INIT_TASK_ID, 
    binding,
  };
}

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto attrs = acc.get_argument<ReduceAttrs>(ATTRS);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  OperatorType op_type = attrs.op_type;

  nonnegative_int reduction_size =
      input.shape.num_elements() / output.shape.num_elements();
  ReducePerDeviceState per_device_state =
      init_kernel(handle,
                  op_type,
                  reduction_size.unwrap_nonnegative(),
                  input.shape,
                  output.shape);
  return DeviceSpecificDeviceStates{
      DeviceSpecific<ReducePerDeviceState>::create(per_device_state)};
}

// Note: forward_kernel only needs ReducePerDeviceState, input, output
OpTaskInvocation forward(ReduceAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<ReducePerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return OpTaskInvocation{
    task_id_t::REDUCE_FWD_TASK_ID, 
    binding,
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ReducePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type = acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reduce] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

OpTaskInvocation backward(ReduceAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
    task_id_t::REDUCE_BWD_TASK_ID, 
    binding,
  };
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ReducePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type = acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reduce] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr());
}

TaskImplFunction get_reduce_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_reduce_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_reduce_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_reduce_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_arg_slot<ReduceAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);

  init.add_return_value<ReducePerDeviceState>();
  return init;
}
OpTaskSignature get_reduce_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_unchecked_arg_slot<ReducePerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  return fwd;
}
OpTaskSignature get_reduce_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_reduce_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(ReduceAttrs const &) {
  return {task_id_t::REDUCE_INIT_TASK_ID,
          task_id_t::REDUCE_FWD_TASK_ID,
          task_id_t::REDUCE_BWD_TASK_ID};
}

}; // namespace FlexFlow
