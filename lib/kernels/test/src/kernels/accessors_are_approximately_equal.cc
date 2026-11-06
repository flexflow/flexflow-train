#include "kernels/accessors_are_approximately_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/local_cuda_allocator.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("accessors_are_approximately_equal") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {1, 2},
            {3, 4},
        },
        cpu_allocator);

    SUBCASE("accessors are the same") {
      CHECK(accessors_are_approximately_equal(lhs, lhs));
    }

    SUBCASE("accessors are very different") {
      GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
          {
              {1, 2},
              {4, 4},
          },
          cpu_allocator);

      CHECK_FALSE(accessors_are_approximately_equal(lhs, rhs));
    }

    SUBCASE("accessors are within the error margin") {
      GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
          {
              {1, 2},
              {3.001, 4},
          },
          cpu_allocator);

      CHECK(accessors_are_approximately_equal(lhs, rhs, 0.0005));
    }
  }
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE(
      "accessors_are_approximately_equal (zeros and relative tolerance)") {
    Allocator allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorR lhs =
        create_1d_accessor_r_with_contents<float>({0, 1000, -2}, allocator);
    GenericTensorAccessorR rhs =
        create_1d_accessor_r_with_contents<float>({0, 1001, -2.001}, allocator);

    CHECK(accessors_are_approximately_equal(lhs, lhs));
    CHECK(accessors_are_approximately_equal(lhs, rhs, 0.01f));
    CHECK_FALSE(accessors_are_approximately_equal(lhs, rhs, 0.0001f));
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("accessors_are_approximately_equal (GPU and CPU accessors)") {
    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorR lhs =
        create_1d_accessor_r_with_contents<float>({0, 1000, -2}, gpu_allocator);
    GenericTensorAccessorR rhs = create_1d_accessor_r_with_contents<float>(
        {0, 1001, -2.001}, cpu_allocator);

    CHECK(accessors_are_approximately_equal(lhs, lhs));
    CHECK(accessors_are_approximately_equal(lhs, rhs, 0.01f));
    CHECK_FALSE(accessors_are_approximately_equal(lhs, rhs, 0.0001f));
  }
}
