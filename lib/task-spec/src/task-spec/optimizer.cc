#include "task-spec/optimizer.h"
#include "kernels/optimizer_kernels.h"
#include "task-spec/profiling.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

enum Slots {
  ATTRS,
  PROFILING,
  HANDLE,
  KERNEL_DEVICE_TYPE,
};

RuntimeTaskInvocation optimizer_attrs_get_update_invocation(
    OptimizerAttrs const &,
    symbolic_forward_tensor_guid_t const &weight,
    symbolic_gradient_tensor_guid_t const &weight_grad,
    std::vector<symbolic_optimizer_tensor_guid_t> const &grad_buffer_tensors) {
  // TODO(@lockshaw)(#pr): 
  NOT_IMPLEMENTED();
}

RuntimeTaskInvocation sgd_update(SGDOptimizerAttrs const &attrs,
                                 symbolic_forward_tensor_guid_t const &weight,
                                 symbolic_gradient_tensor_guid_t const &weight_grad,
                                 symbolic_optimizer_tensor_guid_t const &sgd_v) {
  RuntimeTaskBinding b;
  b.bind(TensorSlotName::WEIGHT, weight);
  b.bind_grad(TensorSlotName::WEIGHT, weight_grad);

  if (attrs.momentum > 0.0f) {
    b.bind_optimizer(TensorSlotName::WEIGHT, OptimizerSlotName::SGD_V, sgd_v);
  }

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  b.bind_arg(HANDLE, ff_handle());
  return RuntimeTaskInvocation{
    task_id_t::SGD_UPD_NCCL_TASK_ID,
    b,
  };
}

static void sgd_update_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<SGDOptimizerAttrs>(ATTRS);
  auto weight_grad = acc.get_tensor_grad<Permissions::RO>(TensorSlotName::WEIGHT);
  auto weight = acc.get_tensor<Permissions::RW>(TensorSlotName::WEIGHT);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  ASSERT(weight.shape == weight_grad.shape);

  ASSERT(get_num_elements(weight_grad.shape.dims).int_from_positive_int() %
             get_num_elements(weight.shape.dims).int_from_positive_int() ==
         0);
  int num_replicas =
      get_num_elements(weight_grad.shape.dims).int_from_positive_int() /
      get_num_elements(weight.shape.dims).int_from_positive_int();

  std::optional<GenericTensorAccessorW> sgd_v = std::nullopt;
  if (attrs.momentum > 0.0f) {
    sgd_v = acc.get_optimizer_tensor<Permissions::RW>(TensorSlotName::WEIGHT, OptimizerSlotName::SGD_V);
    ASSERT(sgd_v.value().shape == weight.shape);
  }

  auto handle = acc.get_argument<device_handle_t>(HANDLE);
  profile(sgd_update_task,
          profiling,
          kernel_device_type,
          "[SGD] update_time = %.2lfms\n",
          handle,
          attrs.lr,
          attrs.momentum,
          attrs.nesterov,
          attrs.weight_decay,
          weight_grad,
          num_replicas,
          weight,
          sgd_v);
}

TaskImplFunction get_sgd_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{sgd_update_task_impl}};
}

RuntimeTaskInvocation adam_update(AdamOptimizerAttrs const &attrs,
                           symbolic_forward_tensor_guid_t const &weight,
                           symbolic_gradient_tensor_guid_t const &weight_grad,
                           symbolic_optimizer_tensor_guid_t const &adam_v,
                           symbolic_optimizer_tensor_guid_t const &adam_m) {

  RuntimeTaskBinding b;
  b.bind(TensorSlotName::WEIGHT, weight);
  b.bind_grad(TensorSlotName::WEIGHT, weight_grad);
  b.bind_optimizer(TensorSlotName::WEIGHT, OptimizerSlotName::ADAM_M, adam_m);
  b.bind_optimizer(TensorSlotName::WEIGHT, OptimizerSlotName::ADAM_V, adam_v);

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  b.bind_arg(HANDLE, ff_handle());

  return RuntimeTaskInvocation{
    task_id_t::ADAM_UPD_NCCL_TASK_ID, 
    b,
  };
}

static void adam_update_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<AdamOptimizerAttrs>(ATTRS);
  auto weight_grad = acc.get_tensor_grad<Permissions::RO>(TensorSlotName::WEIGHT);
  auto weight = acc.get_tensor<Permissions::RW>(TensorSlotName::WEIGHT);
  auto v_tensor = acc.get_optimizer_tensor<Permissions::RW>(TensorSlotName::WEIGHT, OptimizerSlotName::ADAM_V);
  auto m_tensor = acc.get_optimizer_tensor<Permissions::RW>(TensorSlotName::WEIGHT, OptimizerSlotName::ADAM_M);

  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  ASSERT(weight.shape == weight_grad.shape);
  int size = get_num_elements(weight_grad.shape.dims).int_from_positive_int();

  ASSERT(get_num_elements(weight_grad.shape.dims).int_from_positive_int() %
             get_num_elements(weight.shape.dims).int_from_positive_int() ==
         0);
  int num_replicas =
      get_num_elements(weight_grad.shape.dims).int_from_positive_int() /
      get_num_elements(weight.shape.dims).int_from_positive_int();

  auto handle = acc.get_argument<device_handle_t>(HANDLE);
  profile(adam_update_task,
          profiling,
          kernel_device_type,
          "[Adam NCCL] update_time = %.2lfms\n",
          handle,
          attrs.alpha_t,
          attrs.beta1,
          attrs.beta2,
          attrs.weight_decay,
          attrs.epsilon,
          weight_grad.get_float_ptr(),
          size,
          num_replicas,
          m_tensor.get_float_ptr(),
          v_tensor.get_float_ptr(),
          weight.get_float_ptr());
}

TaskImplFunction get_adam_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{adam_update_task_impl}};
}

RuntimeTaskInvocation get_update_invocation(
    OptimizerAttrs const &attrs,
    symbolic_forward_tensor_guid_t const &weight,
    symbolic_gradient_tensor_guid_t const &weight_grad,
    std::vector<symbolic_optimizer_tensor_guid_t> const &grad_buffer_tensors) {
  return attrs.visit<RuntimeTaskInvocation>(
      overload{[&](SGDOptimizerAttrs const &s) {
                 return sgd_update(
                     s, weight, weight_grad, get_only(grad_buffer_tensors));
               },
               [&](AdamOptimizerAttrs const &s) {
                 return adam_update(s,
                                    weight,
                                    weight_grad,
                                    grad_buffer_tensors.at(0),
                                    grad_buffer_tensors.at(1));
               }});
}

TaskImplFunction get_update_task_impl(OptimizerAttrs const &attrs) {
  return attrs.visit<TaskImplFunction>(overload{
      [&](SGDOptimizerAttrs const &) { return get_sgd_update_task_impl(); },
      [&](AdamOptimizerAttrs const &) { return get_adam_update_task_impl(); }});
}

} // namespace FlexFlow
