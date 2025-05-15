#include "internal/test_utils.h"
#include <doctest/doctest.h>
#include "kernels/compare_tensor_accessors.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compare_tensor_accessors_lt") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_3d_accessor_r_with_contents<float>(
        {
          {
            {1, 3, 2},
            {4, 2, 1},
          },
          {
            {3, 3, 6},
            {2, 1, 5},
          },
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_3d_accessor_r_with_contents<float>(
        {
          {
            {2, 3, 3},
            {5, 1, 0},
          },
          {
            {1, 5, 4},
            {2, 1, 5},
          },
        },
        cpu_allocator);

    GenericTensorAccessorW result = compare_tensor_accessors_lt(lhs, rhs, cpu_allocator);
    GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<bool>(
        {
          {
            {true, false, true},
            {true, false, false},
          },
          {
            {false, true, false},
            {false, false, false},
          },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }
}
