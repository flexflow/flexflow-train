#include "task-spec/ops/embedding.h"
#include "kernels/embedding_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Embedding;

enum Slots { INPUT, WEIGHT, OUTPUT, ATTRS, PROFILING, KERNEL_DEVICE_TYPE };

OpTaskInvocation forward(EmbeddingAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0_n));
  b.bind(WEIGHT, weight_tensor(0_n));
  b.bind(OUTPUT, output_tensor(0_n));

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
    task_id_t::EMBED_FWD_TASK_ID, 
    b,
  };
}

OpTaskInvocation backward(EmbeddingAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
    task_id_t::EMBED_BWD_TASK_ID, 
    b,
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  EmbeddingAttrs attrs = acc.get_argument<EmbeddingAttrs>(ATTRS);
  DeviceType kernel_device_type = acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Embedding] forward_time = {:.2lf}ms\n",
                 input,
                 output,
                 weight,
                 input.data_type,
                 output.data_type,
                 attrs.aggr,
                 input.shape.num_dims().unwrap_nonnegative(),
                 output.shape.num_dims().unwrap_nonnegative(),
                 input.shape.at(legion_dim_t{1_n}).int_from_positive_int());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RW>(WEIGHT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  EmbeddingAttrs attrs = acc.get_argument<EmbeddingAttrs>(ATTRS);
  DeviceType kernel_device_type = acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Embedding] backward_time = {:.2lf}ms\n",
                 output,
                 input,
                 weight_grad,
                 output.data_type,
                 input.data_type,
                 attrs.aggr,
                 input.shape.num_dims().unwrap_nonnegative(),
                 output.shape.num_dims().unwrap_nonnegative(),
                 input.shape.at(ff_dim_t{0_n}).int_from_positive_int());
}

TaskImplFunction get_embedding_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_embedding_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_embedding_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_input_slot(OUTPUT);
  fwd.add_input_slot(WEIGHT);

  fwd.add_arg_slot<EmbeddingAttrs>(ATTRS);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<ProfilingSettings>(KERNEL_DEVICE_TYPE);

  return fwd;
}

OpTaskSignature get_embedding_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_embedding_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(EmbeddingAttrs const &) {
  return {task_id_t::EMBED_FWD_TASK_ID, task_id_t::EMBED_BWD_TASK_ID};
}

} // namespace FlexFlow
