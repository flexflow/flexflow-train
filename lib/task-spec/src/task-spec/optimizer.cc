#include "task-spec/optimizer.h"
#include "kernels/optimizer_kernels.h"
#include "task-spec/profiling.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"

namespace FlexFlow {

enum Slots {
  ATTRS,
  WEIGHT,
  WEIGHT_GRAD,
  SGD_V,
  PROFILING,
  ADAM_M,
  ADAM_V,
  HANDLE,
  KERNEL_DEVICE_TYPE,
};

TaskSignature get_sgd_update_signature() {
  TaskSignature sig = make_empty_task_signature();
  add_slot(sig, WEIGHT, TensorType::FORWARD);
  add_slot(sig, WEIGHT_GRAD, TensorType::GRADIENT);
  add_slot(sig, SGD_V, TensorType::OPTIMIZER);

  add_arg_slot<SGDOptimizerAttrs>(sig, ATTRS);
  add_arg_slot<ProfilingSettings>(sig, PROFILING);
  add_arg_slot<DeviceType>(sig, KERNEL_DEVICE_TYPE);
  add_unchecked_arg_slot<PerDeviceFFHandle>(
      sig, HANDLE); // how to deal with removal of ParamSync?

  // if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
  //   add_unchecked_arg_slot<PerDeviceFFHandle>(sig, HANDLE);
  // }
  return sig;
}

TaskInvocation sgd_update(SGDOptimizerAttrs const &attrs,
                          forward_tensor_guid_t const &weight,
                          gradient_tensor_guid_t const &weight_grad,
                          optimizer_tensor_guid_t const &sgd_v) {
  TaskBinding b;
  b.bind(WEIGHT, weight);
  b.bind_grad(WEIGHT_GRAD, weight_grad);

  if (attrs.momentum > 0.0f) {
    b.bind_optimizer(SGD_V, sgd_v);
  }
  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  b.bind_arg(HANDLE, ff_handle());
  return TaskInvocation{task_id_t::SGD_UPD_NCCL_TASK_ID,
                        b}; // how to deal with removal of ParamSync?

  // if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
  //   b.bind_arg(HANDLE, ff_handle());
  //   return TaskInvocation{task_id_t::SGD_UPD_NCCL_TASK_ID, b};
  // } else {
  //   return TaskInvocation{task_id_t::SGD_UPD_PS_TASK_ID, b};
  // }
}

static void sgd_update_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<SGDOptimizerAttrs>(ATTRS);
  auto weight_grad = acc.get_tensor_grad<Permissions::RO>(WEIGHT_GRAD);
  auto weight = acc.get_tensor<Permissions::RW>(WEIGHT);
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
    sgd_v = acc.get_optimizer_tensor<Permissions::RW>(SGD_V);
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
          sgd_v); // how to deal with removal of ParamSync?

  // if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
  //   auto handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  //   profile(sgd_nccl_update_task_gpu,
  //           profiling,
  //           "[SGD NCCL] update_time = %.2lfms\n",
  //           attrs.lr,
  //           attrs.momentum,
  //           attrs.nesterov,
  //           attrs.weight_decay,
  //           handle,
  //           weight_grad.get_float_ptr(),
  //           size,
  //           weight.get_float_ptr(),
  //           sgd_v_ptr);

  // } else {
  //   profile(sgd_ps_update_task_gpu,
  //           profiling,
  //           "[SGD PS] update_time = %.2lfms\n",
  //           attrs.lr,
  //           attrs.momentum,
  //           attrs.nesterov,
  //           attrs.weight_decay,
  //           weight_grad.get_float_ptr(),
  //           size,
  //           num_replicas,
  //           weight.get_float_ptr(),
  //           sgd_v_ptr);
  // }
}

TaskImplFunction get_sgd_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{sgd_update_task_impl}};
}

TaskSignature get_adam_update_signature() {
  TaskSignature sig = make_empty_task_signature();
  add_slot(sig, WEIGHT, TensorType::FORWARD);
  add_slot(sig, WEIGHT_GRAD, TensorType::GRADIENT);
  add_slot(sig, ADAM_V, TensorType::OPTIMIZER);
  add_slot(sig, ADAM_M, TensorType::OPTIMIZER);

  add_arg_slot<AdamOptimizerAttrs>(sig, ATTRS);
  add_arg_slot<ProfilingSettings>(sig, PROFILING);
  add_arg_slot<DeviceType>(sig, KERNEL_DEVICE_TYPE);
  add_unchecked_arg_slot<PerDeviceFFHandle>(
      sig, HANDLE); // how to deal with removal of ParamSync?
  // if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
  //   add_unchecked_arg_slot<PerDeviceFFHandle>(sig, HANDLE);
  // }
  return sig;
}

TaskInvocation adam_update(AdamOptimizerAttrs const &attrs,
                           forward_tensor_guid_t const &weight,
                           gradient_tensor_guid_t const &weight_grad,
                           optimizer_tensor_guid_t const &adam_v,
                           optimizer_tensor_guid_t const &adam_m) {
  TaskBinding b;
  b.bind(WEIGHT, weight);
  b.bind_grad(WEIGHT_GRAD, weight_grad);
  b.bind_optimizer(ADAM_M, adam_m);
  b.bind_optimizer(ADAM_V, adam_v);
  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  b.bind_arg(HANDLE, ff_handle());
  return TaskInvocation{task_id_t::ADAM_UPD_NCCL_TASK_ID,
                        b}; // how to deal with removal of ParamSync?

  // if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
  //   b.bind_arg(HANDLE, ff_handle());
  //   return TaskInvocation{task_id_t::ADAM_UPD_NCCL_TASK_ID, b};
  // } else {
  //   return TaskInvocation{task_id_t::ADAM_UPD_PS_TASK_ID, b};
  // }
}

static void adam_update_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<AdamOptimizerAttrs>(ATTRS);
  auto weight_grad = acc.get_tensor_grad<Permissions::RO>(WEIGHT_GRAD);
  auto weight = acc.get_tensor<Permissions::RW>(WEIGHT);
  auto v_tensor = acc.get_optimizer_tensor<Permissions::RW>(ADAM_V);
  auto m_tensor = acc.get_optimizer_tensor<Permissions::RW>(ADAM_M);

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
          weight.get_float_ptr()); // how to deal with removal of ParamSync?

  // if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
  //   auto handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  //   profile(adam_nccl_update_task_gpu,
  //           profiling,
  //           "[Adam NCCL] update_time = %.2lfms\n",
  //           attrs.alpha_t,
  //           attrs.beta1,
  //           attrs.beta2,
  //           attrs.weight_decay,
  //           attrs.epsilon,
  //           size,
  //           handle,
  //           weight_grad.get_float_ptr(),
  //           m_tensor.get_float_ptr(),
  //           v_tensor.get_float_ptr(),
  //           weight.get_float_ptr());
  // } else {
  //   profile(adam_ps_update_task_gpu,
  //           profiling,
  //           "[Adam NCCL] update_time = %.2lfms\n",
  //           attrs.alpha_t,
  //           attrs.beta1,
  //           attrs.beta2,
  //           attrs.weight_decay,
  //           attrs.epsilon,
  //           size,
  //           num_replicas,
  //           weight_grad.get_float_ptr(),
  //           m_tensor.get_float_ptr(),
  //           v_tensor.get_float_ptr(),
  //           weight.get_float_ptr());
  // }
}

TaskImplFunction get_adam_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{adam_update_task_impl}};
}

TaskSignature get_update_signature(OptimizerAttrs const &attrs) {
  return attrs.visit<TaskSignature>(overload{
      [&](SGDOptimizerAttrs const &) { return get_sgd_update_signature(); },
      [&](AdamOptimizerAttrs const &) { return get_adam_update_signature(); }});
}

TaskInvocation get_update_invocation(
    OptimizerAttrs const &attrs,
    forward_tensor_guid_t const &weight,
    gradient_tensor_guid_t const &weight_grad,
    std::vector<optimizer_tensor_guid_t> const &grad_buffer_tensors) {
  return attrs.visit<TaskInvocation>(
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
