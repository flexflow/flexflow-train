#include <doctest/doctest.h>
#include "internal/test_utils.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/create_accessor_with_contents.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/repeat_element.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_accessor_scale_by_constant") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
          {
              {3, 3, 6},
              {0, -1, 0.75},
          },
        cpu_allocator);

    GenericTensorAccessorW result = tensor_accessor_scale_by_constant(input, -2.0, cpu_allocator);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
          {
              {-6, -6, -12},
              {0, 2, -1.5},
          },
          cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result=", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_broadcast") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
          {
              {3},
              {-0.5}, 
              {6},
          },
        cpu_allocator);

    TensorDims output_dims = TensorDims{
      FFOrdered{4_p, 1_p, 3_p, 2_p},
    };

    GenericTensorAccessorW result = tensor_accessor_broadcast(input, output_dims, cpu_allocator);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
          repeat_element(4_n, 
            std::vector<std::vector<std::vector<float>>>{
              std::vector<std::vector<float>>{
                repeat_element<float>(2_n, 3.0),
                repeat_element<float>(2_n, -0.5), 
                repeat_element<float>(2_n, 6.0),
              }
            }),
          cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result=", format_accessor_w_contents(result)));
  }
}
