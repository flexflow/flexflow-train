#include "task-spec/ops/element_unary.h"
#include "kernels/element_unary_kernels.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

// declare Legion names

using namespace FlexFlow::Kernels::ElementUnary;

enum Slots {
  INPUT,
  INPUT_SHAPE,
  OUTPUT,
  OUTPUT_SHAPE,
  ATTRS,
  HANDLE,
  PROFILING,
  PER_DEVICE_STATE,
  KERNEL_DEVICE_TYPE,
};

/* ElementUnary */
OpTaskInvocation init(ElementUnaryAttrs const &attrs) {
  OpTaskBinding b;

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  b.bind_arg(INPUT_SHAPE, input_parallel_tensor_shape(0_n));
  b.bind_arg(OUTPUT_SHAPE, output_parallel_tensor_shape(0_n));

  return OpTaskInvocation{
      task_id_t::ELEMENTUNARY_INIT_TASK_ID,
      b,
  };
}

OpTaskInvocation forward(ElementUnaryAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0_n));
  b.bind(OUTPUT, output_tensor(0_n));
  b.bind_arg(ATTRS, attrs);

  b.bind_arg(HANDLE, ff_handle());
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  b.bind_arg(PER_DEVICE_STATE,
             per_device_op_state<std::optional<ElementUnaryPerDeviceState>>());

  return OpTaskInvocation{
      task_id_t::ELEMENTUNARY_FWD_TASK_ID,
      b,
  };
}

OpTaskInvocation backward(ElementUnaryAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::ELEMENTUNARY_BWD_TASK_ID,
      b,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {

  auto attrs = acc.get_argument<ElementUnaryAttrs>(ATTRS);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  ParallelTensorShape input_shape =
      acc.get_argument<ParallelTensorShape>(INPUT_SHAPE);
  ParallelTensorShape output_shape =
      acc.get_argument<ParallelTensorShape>(OUTPUT_SHAPE);

  std::optional<ElementUnaryPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  get_piece_shape(input_shape),
                  get_piece_shape(output_shape),
                  attrs);

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<ElementUnaryAttrs>(ATTRS);

  auto handle = acc.get_argument<device_handle_t>(HANDLE);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementUnary] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 attrs,
                 handle,
                 input,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  auto const &attrs = acc.get_argument<ElementUnaryAttrs>(ATTRS);
  auto handle = acc.get_argument<device_handle_t>(HANDLE);

  auto per_device_state =
      acc.get_argument<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementUnary] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 attrs,
                 handle,
                 output,
                 output_grad,
                 input,
                 input_grad);
}

TaskImplFunction get_element_unary_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_element_unary_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_element_unary_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_element_unary_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<ParallelTensorShape>(INPUT_SHAPE);
  init.add_arg_slot<ElementUnaryAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_unchecked_arg_slot<device_handle_t>(HANDLE);

  init.add_return_value<ElementUnaryPerDeviceState>();

  return init;
}

OpTaskSignature get_element_unary_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_unchecked_arg_slot<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);

  return fwd;
}

OpTaskSignature get_element_unary_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_element_unary_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(ElementUnaryAttrs const &) {
  return {task_id_t::ELEMENTUNARY_INIT_TASK_ID,
          task_id_t::ELEMENTUNARY_FWD_TASK_ID,
          task_id_t::ELEMENTUNARY_BWD_TASK_ID};
}

} // namespace FlexFlow
