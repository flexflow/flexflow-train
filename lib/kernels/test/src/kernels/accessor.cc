#include "kernels/accessor.h"
#include "kernels/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/local_cpu_allocator.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("calculate_accessor_offset") {
    SUBCASE("one dimension") {
      std::vector<nonnegative_int> indices = {4_n};
      ArrayShape shape = ArrayShape{
          std::vector{
              13_p,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 4_n;

      CHECK(result == correct);
    }

    SUBCASE("multiple dimensions") {
      std::vector<nonnegative_int> indices = {2_n, 4_n};
      ArrayShape shape = ArrayShape{
          std::vector{
              6_p,
              5_p,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 2_n * 5_n + 4_n;

      CHECK(result == correct);
    }

    SUBCASE("zero dimensions") {
      std::vector<nonnegative_int> indices = {};
      ArrayShape shape = ArrayShape{std::vector<positive_int>{}};

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("index and shape dimensions do not match") {
      std::vector<nonnegative_int> indices = {1_n, 2_n, 4_n};
      ArrayShape shape = ArrayShape{
          std::vector<positive_int>{
              6_p,
              5_p,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }

    SUBCASE("out of bounds index") {
      std::vector<nonnegative_int> indices = {2_n, 5_n};
      ArrayShape shape = ArrayShape{
          std::vector<positive_int>{
              6_p,
              5_p,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }
  }

  TEST_CASE("accessor_get_only_value") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("returns the value if the accessor only contains one value") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
              },
          },
          cpu_allocator);

      float result = accessor_get_only_value<DataType::FLOAT>(input);
      float correct = 12;

      CHECK(result == correct);
    }

    SUBCASE("throws an error if the requested type does not match the type in "
            "the accessor") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
              },
          },
          cpu_allocator);

      CHECK_THROWS(accessor_get_only_value<DataType::DOUBLE>(input));
    }

    SUBCASE("throws an error if the accessor contains multiple values") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
                  {12},
              },
          },
          cpu_allocator);

      CHECK_THROWS(accessor_get_only_value<DataType::FLOAT>(input));
    }
  }
}
