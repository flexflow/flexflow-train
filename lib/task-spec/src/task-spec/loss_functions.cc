/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op-attrs/ops/loss_functions.h"
#include "kernels/loss_function_kernels.h"
#include "task-spec/loss_functions.h"
#include "task-spec/profiling.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

enum Slots { LOGIT, LABEL, LOGIT_GRAD, ATTRS, PROFILING, KERNEL_DEVICE_TYPE };

TaskSignature get_loss_bwd_signature() {
  TaskSignature sig = make_empty_task_signature();
  add_slot(sig, LOGIT, TensorType::FORWARD);
  add_slot(sig, LABEL, TensorType::LOSS);
  add_slot(sig, LOGIT_GRAD, TensorType::GRADIENT);

  add_arg_slot<LossAttrs>(sig, ATTRS);
  add_arg_slot<ProfilingSettings>(sig, PROFILING);
  add_arg_slot<DeviceType>(sig, KERNEL_DEVICE_TYPE);
  return sig;
}

TaskInvocation backward(LossAttrs const &attrs,
                        forward_tensor_guid_t logit,
                        gradient_tensor_guid_t logit_grad,
                        loss_tensor_guid_t label) {
  TaskBinding b;
  b.bind(LOGIT, logit);
  b.bind_loss(LABEL, label);
  b.bind_grad(LOGIT_GRAD, logit_grad);

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return TaskInvocation{task_id_t::LOSS_BWD_TASK_ID, b};
}

static void backward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<LossAttrs>(ATTRS);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto kernel_device_type = acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto logit_grad = acc.get_tensor_grad<Permissions::RW>(LOGIT_GRAD);
  auto logit = acc.get_tensor<Permissions::RO>(LOGIT);
  auto label = acc.get_loss_tensor<Permissions::RO>(LABEL);

  int batch_size = dim_at_idx(logit.shape.dims, legion_dim_t{1_n}).int_from_positive_int();
  // assuming logit shape is [batch dim, num classes]

  LossFunction loss_type = get_loss_function(attrs);
  float scale_factor = 1.0f / batch_size;
  if (loss_type == LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE) {
    ASSERT(get_num_elements(logit.shape.dims) == get_num_elements(label.shape.dims));
    scale_factor = 2.0f / get_num_elements(logit.shape.dims).int_from_positive_int();
  }

  if (loss_type == LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY) {
    // label shape is [batch dim, 1]
    auto scce_attrs = attrs.get<SparseCategoricalCrossEntropyLossAttrs>();
    size_t ndim = get_num_dims(logit.shape.dims).unwrap_nonnegative();
    int num_classes = dim_at_idx(logit.shape.dims, legion_dim_t{0_n}).int_from_positive_int();
    ASSERT(logit_grad.shape == logit.shape);
    int k = 1;
    if (scce_attrs.replace_labels) {
      k = dim_at_idx(logit.shape.dims, legion_dim_t{nonnegative_int{ndim - 1}})
              .int_from_positive_int() /
          dim_at_idx(label.shape.dims, legion_dim_t{nonnegative_int{ndim - 1}})
              .int_from_positive_int(); // TODO FIXME something seems wrong
                                        // here, isn't the numerator guaranteed
                                        // to be 1?
                                        // <--- this is not the case because of
                                        // the potential parallel dim
    }
    ASSERT(slice_tensor_dims(label.shape.dims, relative_ff_dim_t{0}, relative_ff_dim_t{-2}) ==
           slice_tensor_dims(logit.shape.dims, relative_ff_dim_t{0}, relative_ff_dim_t{-2}));
    ASSERT(k * dim_at_idx(label.shape.dims, legion_dim_t{nonnegative_int{ndim - 1}})
                   .int_from_positive_int() ==
           dim_at_idx(logit.shape.dims, legion_dim_t{nonnegative_int{ndim - 1}})
               .int_from_positive_int());
    ASSERT(dim_at_idx(label.shape.dims, legion_dim_t(0_n)).int_from_positive_int() == 1);

    profile(sparse_categorical_crossentropy_loss_backward_kernel,
            profiling,
            kernel_device_type,
            "[SparseCategoricalCrossEntropyLoss] backward_time = %.2lfms\n",
            get_float_ptr(logit_grad),
            get_float_ptr(logit),
            reinterpret_cast<int const *>(get_float_ptr(label)),
            get_num_elements(logit.shape.dims).int_from_positive_int(),
            get_num_elements(logit_grad.shape.dims).int_from_positive_int(),
            batch_size,
            num_classes,
            k,
            scale_factor);
  } else {
    ASSERT(logit.shape == label.shape);
    ASSERT(logit_grad.shape == logit.shape);
    int num_channels =
        dim_at_idx(logit.shape.dims, legion_dim_t{0_n}).int_from_positive_int();
    switch (loss_type) {
      case LossFunction::CATEGORICAL_CROSSENTROPY: {
        profile(categorical_crossentropy_loss_backward_kernel,
                profiling,
                kernel_device_type,
                "[CategoricalCrossEntropyLoss] backward_time = %.2lfms\n",
                logit_grad,
                logit,
                label,
                scale_factor);
        break;
      }
      case LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE: {
        profile(mean_squared_error_avg_loss_backward_kernel,
                profiling,
                kernel_device_type,
                "[MeanSquaredErrorAvgLoss] backward_time = %.2lfms\n",
                get_float_ptr(logit_grad),
                get_float_ptr(logit),
                get_float_ptr(label),
                get_num_elements(logit.shape.dims).int_from_positive_int(),
                get_num_elements(logit_grad.shape.dims).int_from_positive_int(),
                scale_factor);
        break;
      }
      case LossFunction::IDENTITY: {
        profile(identity_loss_backward_kernel,
                profiling,
                kernel_device_type,
                "[IdentityLoss] backward_time = %.2lfms\n",
                get_float_ptr(logit_grad),
                get_float_ptr(logit),
                get_num_elements(logit.shape.dims).int_from_positive_int(),
                get_num_elements(logit_grad.shape.dims).int_from_positive_int(),
                scale_factor);
        break;
      }
      default:
        PANIC(fmt::format(
            "Unsupported loss function {}. Please report this as an issue.",
            loss_type));
    }
  }
}

TaskImplFunction get_loss_bwd_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow
