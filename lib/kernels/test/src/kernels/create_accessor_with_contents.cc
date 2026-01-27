#include "kernels/create_accessor_with_contents.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("create_1d_accessor_w_with_contents") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW result =
        create_1d_accessor_w_with_contents<float>({1, 4, 1, 2}, cpu_allocator);

    auto at = [&](nonnegative_int c) -> float {
      return result.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{c}});
    };

    CHECK(at(0_n) == 1);
    CHECK(at(1_n) == 4);
    CHECK(at(2_n) == 1);
    CHECK(at(3_n) == 2);
  }

  TEST_CASE("create_2d_accessor_w_with_contents") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW result = create_2d_accessor_w_with_contents<float>(
        {
            {1, 4, 2},
            {2, 2, 7},
        },
        cpu_allocator);

    auto at = [&](nonnegative_int r, nonnegative_int c) -> float {
      return result.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{r, c}});
    };

    CHECK(at(0_n, 0_n) == 1);
    CHECK(at(0_n, 1_n) == 4);
    CHECK(at(0_n, 2_n) == 2);
    CHECK(at(1_n, 0_n) == 2);
    CHECK(at(1_n, 1_n) == 2);
    CHECK(at(1_n, 2_n) == 7);
  }

  TEST_CASE("create_3d_accessor_w_with_contents") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW result = create_3d_accessor_w_with_contents<float>(
        {
            {
                {1, 4},
                {2, 3},
                {7, 2},
            },
            {
                {9, 3},
                {4, 5},
                {0, 2},
            },
        },
        cpu_allocator);

    auto at =
        [&](nonnegative_int s, nonnegative_int r, nonnegative_int c) -> float {
      return result.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{s, r, c}});
    };

    CHECK(at(0_n, 0_n, 0_n) == 1);
    CHECK(at(0_n, 0_n, 1_n) == 4);
    CHECK(at(0_n, 1_n, 0_n) == 2);
    CHECK(at(0_n, 1_n, 1_n) == 3);
    CHECK(at(0_n, 2_n, 0_n) == 7);
    CHECK(at(0_n, 2_n, 1_n) == 2);
    CHECK(at(1_n, 0_n, 0_n) == 9);
    CHECK(at(1_n, 0_n, 1_n) == 3);
    CHECK(at(1_n, 1_n, 0_n) == 4);
    CHECK(at(1_n, 1_n, 1_n) == 5);
    CHECK(at(1_n, 2_n, 0_n) == 0);
    CHECK(at(1_n, 2_n, 1_n) == 2);
  }

  TEST_CASE("create_4d_accessor_w_with_contents") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW result = create_4d_accessor_w_with_contents<float>(
        {
            {
                {
                    {2, 3},
                    {7, 2},
                },
                {
                    {4, 5},
                    {0, 2},
                },
            },
            {
                {
                    {9, 6},
                    {1, 2},
                },
                {
                    {8, 7},
                    {3, 8},
                },
            },
        },
        cpu_allocator);

    auto at = [&](nonnegative_int s1,
                  nonnegative_int s2,
                  nonnegative_int r,
                  nonnegative_int c) -> float {
      return result.at<DataType::FLOAT>(
          TensorDimsCoord{FFOrdered{s1, s2, r, c}});
    };

    CHECK(at(0_n, 0_n, 0_n, 0_n) == 2);
    CHECK(at(0_n, 0_n, 0_n, 1_n) == 3);
    CHECK(at(0_n, 0_n, 1_n, 0_n) == 7);
    CHECK(at(0_n, 0_n, 1_n, 1_n) == 2);
    CHECK(at(0_n, 1_n, 0_n, 0_n) == 4);
    CHECK(at(0_n, 1_n, 0_n, 1_n) == 5);
    CHECK(at(0_n, 1_n, 1_n, 0_n) == 0);
    CHECK(at(0_n, 1_n, 1_n, 1_n) == 2);
    CHECK(at(1_n, 0_n, 0_n, 0_n) == 9);
    CHECK(at(1_n, 0_n, 0_n, 1_n) == 6);
    CHECK(at(1_n, 0_n, 1_n, 0_n) == 1);
    CHECK(at(1_n, 0_n, 1_n, 1_n) == 2);
    CHECK(at(1_n, 1_n, 0_n, 0_n) == 8);
    CHECK(at(1_n, 1_n, 0_n, 1_n) == 7);
    CHECK(at(1_n, 1_n, 1_n, 0_n) == 3);
    CHECK(at(1_n, 1_n, 1_n, 1_n) == 8);
  }
}
