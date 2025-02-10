#include "doctest/doctest.h"
#include "kernels/concat_kernels.h"
#include "test_utils.h"
#include "utils/containers/repeat.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test concat kernel forward and backward") {
    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};
    Allocator allocator = create_local_cuda_memory_allocator();

    const nonnegative_int num_inputs = 4_n;

    SUBCASE("forward_kernel") {
      auto run_forward_test = [&](nonnegative_int input_rows,
                                  nonnegative_int input_cols,
                                  TensorShape output_shape,
                                  ff_dim_t concat_axis) {
        TensorShape input_shape = make_tensor_shape_from_legion_dims(
            {input_rows, input_cols}, DataType::FLOAT);

        std::vector<GenericTensorAccessorR> input_accessors =
            repeat(num_inputs, [&]() {
              return create_random_filled_accessor_r(input_shape, allocator);
            });

        GenericTensorAccessorW output_accessor =
            allocator.allocate_tensor(output_shape);

        Kernels::Concat::forward_kernel(managed_stream.raw_stream(),
                                        output_accessor,
                                        input_accessors,
                                        concat_axis);

        CHECK(contains_non_zero(output_accessor));
      };

      SUBCASE("test forward concat, axis = 0") {
        nonnegative_int input_rows = 2_n;
        nonnegative_int input_cols = 4_n;
        TensorShape output_shape = make_tensor_shape_from_legion_dims(
            {num_inputs * input_rows, input_cols}, DataType::FLOAT);
        run_forward_test(input_rows, input_cols, output_shape, ff_dim_t{0_n});
      }

      SUBCASE("test forward concat, axis = 1") {
        nonnegative_int input_rows = 4_n;
        nonnegative_int input_cols = 2_n;
        TensorShape output_shape = make_tensor_shape_from_legion_dims(
            {input_rows, num_inputs * input_cols}, DataType::FLOAT);
        run_forward_test(input_rows, input_cols, output_shape, ff_dim_t{1_n});
      }
    }

    SUBCASE("backward_kernel") {
      auto run_backward_test = [&](nonnegative_int input_rows,
                                   nonnegative_int input_cols,
                                   TensorShape output_shape,
                                   ff_dim_t concat_axis) {
        TensorShape input_shape = make_tensor_shape_from_legion_dims(
            {input_rows, input_cols}, DataType::FLOAT);

        GenericTensorAccessorR output_grad_accessor =
            create_random_filled_accessor_r(output_shape, allocator);

        std::vector<GenericTensorAccessorW> input_grad_accessors =
            repeat(num_inputs, [&]() {
              return create_zero_filled_accessor_w(input_shape, allocator);
            });

        Kernels::Concat::backward_kernel(managed_stream.raw_stream(),
                                         output_grad_accessor,
                                         input_grad_accessors,
                                         concat_axis);

        for (auto &accessor : input_grad_accessors) {
          CHECK(contains_non_zero(accessor));
        }
      };

      SUBCASE("test backward concat, axis = 0") {
        nonnegative_int input_rows = 2_n;
        nonnegative_int input_cols = 4_n;
        TensorShape output_shape = make_tensor_shape_from_legion_dims(
            {num_inputs * input_rows, input_cols}, DataType::FLOAT);
        run_backward_test(input_rows, input_cols, output_shape, ff_dim_t{0_n});
      }

      SUBCASE("test backward concat, axis = 1") {
        nonnegative_int input_rows = 4_n;
        nonnegative_int input_cols = 2_n;
        TensorShape output_shape = make_tensor_shape_from_legion_dims(
            {input_rows, num_inputs * input_cols}, DataType::FLOAT);
        run_backward_test(input_rows, input_cols, output_shape, ff_dim_t{1_n});
      }
    }
  }
}
