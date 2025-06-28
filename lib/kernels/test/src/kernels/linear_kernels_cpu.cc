#include "internal/test_utils.h"
#include <doctest/doctest.h>
#include "kernels/linear_kernels_cpu.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cpu_forward_kernel (Linear)") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
          {
              {3, 3, 6},
              {2, 1, 5},
              {1, 2, -2},
              {8, 0.5, -3},
          },
        cpu_allocator);

    GenericTensorAccessorR filter = create_2d_accessor_r_with_contents<float>(
          {
              {1.0f, 0.5f},
              {2.0f, 4.0f},
              {1.5f, -1.0f},
          },
          cpu_allocator);

    GenericTensorAccessorR bias = create_1d_accessor_r_with_contents<float>(
        {3.0, -1.0}, cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{4_p, 2_p}},
            DataType::FLOAT,
        },
        cpu_allocator);

    Kernels::Linear::cpu_forward_kernel(input, result, filter, bias);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
          {21.0f, 6.5f},
          {14.5f, -1.0f},
          {5.0f, 9.5f},
          {7.5f, 8.0f},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result=", format_accessor_w_contents(result)));
  }
}
