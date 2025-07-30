#include "kernels/batch_matmul_kernels.h"
#include "kernels/batch_matmul_kernels_cpu.h"
#include "kernels/batch_matmul_kernels_gpu.h"
#include "utils/containers/require_same.h"

namespace FlexFlow::Kernels::BatchMatmul {

static std::tuple<positive_int, positive_int, positive_int, positive_int>
    get_params(TensorDims const &input_a_dims,
               TensorDims const &input_b_dims,
               TensorDims const &output_dims) {
  positive_int m = require_same(dim_at_idx(input_b_dims, relative_ff_dim_t{-1}),
                                dim_at_idx(output_dims, relative_ff_dim_t{-1}));

  positive_int n = require_same(dim_at_idx(input_a_dims, relative_ff_dim_t{-2}),
                                dim_at_idx(output_dims, relative_ff_dim_t{-2}));

  positive_int k =
      require_same(dim_at_idx(input_a_dims, relative_ff_dim_t{-1}),
                   dim_at_idx(input_b_dims, relative_ff_dim_t{-2}));

  TensorDims leading_dims = require_same(
      slice_tensor_dims(
          input_a_dims, relative_ff_dim_t{0}, relative_ff_dim_t{-2}),
      slice_tensor_dims(
          input_b_dims, relative_ff_dim_t{0}, relative_ff_dim_t{-2}));

  positive_int batch = get_num_elements(leading_dims);

  return {m, n, k, batch};
}

void forward_kernel(device_stream_t const &stream,
                    device_handle_t const &handle,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const &input_a,
                    GenericTensorAccessorR const &input_b,
                    positive_int seq_length,
                    std::optional<positive_int> a_seq_length_dim,
                    std::optional<positive_int> b_seq_length_dim) {

  auto [m, n, k, batch] =
      get_params(input_a.shape.dims, input_b.shape.dims, output.shape.dims);

  auto get_raw_seq_len = [](std::optional<positive_int> seq_len) -> int {
    return transform(seq_len,
                     [](positive_int x) { return x.int_from_positive_int(); })
        .value_or(-1);
  };

  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*output_ptr=*/output.get_float_ptr(),
        /*a_input_ptr=*/input_a.get_float_ptr(),
        /*b_input_ptr=*/input_b.get_float_ptr(),
        /*m=*/m.int_from_positive_int(),
        /*n=*/n.int_from_positive_int(),
        /*k=*/k.int_from_positive_int(),
        /*batch=*/batch.int_from_positive_int(),
        /*seq_length=*/seq_length.int_from_positive_int(),
        /*a_seq_length_dim=*/get_raw_seq_len(a_seq_length_dim),
        /*b_seq_length_dim=*/get_raw_seq_len(b_seq_length_dim));
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    cpu_forward_kernel(
        /*output=*/output,
        /*input_a=*/input_a,
        /*input_b=*/input_b,
        /*seq_length=*/seq_length,
        /*a_seq_length_dim=*/a_seq_length_dim,
        /*b_seq_length_dim=*/b_seq_length_dim);
  }
}

void backward_kernel(device_stream_t const &stream,
                     device_handle_t const &handle,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input_a,
                     GenericTensorAccessorW const &input_a_grad,
                     GenericTensorAccessorR const &input_b,
                     GenericTensorAccessorW const &input_b_grad) {
  TensorShape input_a_shape = require_same(input_a.shape, input_a_grad.shape);
  TensorShape input_b_shape = require_same(input_b.shape, input_b_grad.shape);
  TensorShape output_shape = require_same(output.shape, output_grad.shape);

  auto [m, n, k, batch] =
      get_params(input_a_shape.dims, input_b_shape.dims, output_shape.dims);

  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*output_ptr=*/output.get_float_ptr(),
        /*output_grad_ptr=*/output_grad.get_float_ptr(),
        /*input_a_ptr=*/input_a.get_float_ptr(),
        /*input_a_grad_ptr=*/input_a_grad.get_float_ptr(),
        /*input_b_ptr=*/input_b.get_float_ptr(),
        /*input_b_grad_ptr=*/input_b_grad.get_float_ptr(),
        /*m=*/m.int_from_positive_int(),
        /*n=*/n.int_from_positive_int(),
        /*k=*/k.int_from_positive_int(),
        /*batch=*/batch.int_from_positive_int());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    cpu_backward_kernel(
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input_a=*/input_a,
        /*input_a_grad=*/input_a_grad,
        /*input_b=*/input_b,
        /*input_b_grad=*/input_b_grad);
  }
}

} // namespace FlexFlow::Kernels::BatchMatmul
