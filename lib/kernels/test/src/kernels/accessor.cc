#include "kernels/accessor.h"
#include "internal/test_utils.h"
#include "kernels/local_cpu_allocator.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("calculate_accessor_offset") {
    SUBCASE("one dimension") {
      std::vector<nonnegative_int> indices = {4_n};
      ArrayShape shape = ArrayShape{
          std::vector<nonnegative_int>{
              13_n,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 4_n;

      CHECK(result == correct);
    }

    SUBCASE("multiple dimensions") {
      std::vector<nonnegative_int> indices = {2_n, 4_n};
      ArrayShape shape = ArrayShape{
          std::vector<nonnegative_int>{
              6_n,
              5_n,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 2_n * 5_n + 4_n;

      CHECK(result == correct);
    }

    SUBCASE("zero dimensions") {
      std::vector<nonnegative_int> indices = {};
      ArrayShape shape = ArrayShape{std::vector<nonnegative_int>{}};

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("index and shape dimensions do not match") {
      std::vector<nonnegative_int> indices = {1_n, 2_n, 4_n};
      ArrayShape shape = ArrayShape{
          std::vector<nonnegative_int>{
              6_n,
              5_n,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }

    SUBCASE("out of bounds index") {
      std::vector<nonnegative_int> indices = {2_n, 5_n};
      ArrayShape shape = ArrayShape{
          std::vector<nonnegative_int>{
              6_n,
              5_n,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }
  }

  TEST_CASE("format_1d_accessor_contents(GenericTensorAccessorR)") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("accessor is 1d") {
      GenericTensorAccessorR accessor =
          create_1d_accessor_r_with_contents({1, 2, 3, 2}, cpu_allocator);

      std::string correct = "[ 1 2 3 2 ]";

      std::string result = format_1d_accessor_contents(accessor);

      CHECK(result == correct);
    }

    SUBCASE("accessor is not 1d") {
      GenericTensorAccessorR accessor = create_2d_accessor_r_with_contents(
          {
              {1, 2, 3},
              {4, 3, 3},
              {1, 1, 5},
          },
          cpu_allocator);

      CHECK_THROWS(format_1d_accessor_contents(accessor));
    }
  }

  TEST_CASE("format_2d_accessor_contents(GenericTensorAccessorR)") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("accessor is 2d") {
      GenericTensorAccessorR accessor = create_2d_accessor_r_with_contents(
          {
              {1, 2, 3},
              {4, 3, 3},
              {1, 1, 5},
          },
          cpu_allocator);

      std::string correct = "[ 1 2 3 ]\n"
                            "[ 4 3 3 ]\n"
                            "[ 1 1 5 ]";

      std::string result = format_2d_accessor_contents(accessor);

      CHECK(result == correct);
    }

    SUBCASE("accessor is not 2d") {
      GenericTensorAccessorR accessor =
          create_1d_accessor_r_with_contents({1, 2, 3}, cpu_allocator);

      CHECK_THROWS(format_2d_accessor_contents(accessor));
    }
  }
}
