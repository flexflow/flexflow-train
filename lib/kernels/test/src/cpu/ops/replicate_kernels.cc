#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/replicate_kernels_cpu.h"
#include "kernels/test_utils.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Replicate::cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input =
        create_1d_accessor_r_with_contents<int32_t>({1, 3, 2}, cpu_allocator);

    TensorShape result_shape = TensorShape{
        TensorDims{FFOrdered{3_p}},
        DataType::INT32,
    };
    GenericTensorAccessorW result =
        create_zero_filled_accessor_w(result_shape, cpu_allocator);

    GenericTensorAccessorR correct = input;

    Kernels::Replicate::cpu_forward_kernel(input, result);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  "result=",
                  format_accessor_w_contents(result));
  }

  TEST_CASE("Replicate::cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR output = create_2d_accessor_r_with_contents<int32_t>(
        {
            {1, 2, 3},
            {4, 3, 3},
            {1, 3, 5},
        },
        cpu_allocator);

    GenericTensorAccessorR correct =
        create_1d_accessor_r_with_contents<int32_t>(
            {1 + 2 + 3, 4 + 3 + 3, 1 + 3 + 5}, cpu_allocator);

    TensorShape result_shape = TensorShape{
        TensorDims{FFOrdered{3_p}},
        DataType::INT32,
    };
    GenericTensorAccessorW result =
        create_zero_filled_accessor_w(result_shape, cpu_allocator);
    Kernels::Replicate::cpu_backward_kernel(output, result, 3);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }
}
