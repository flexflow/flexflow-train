#include "kernels/accessor.h"
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
}
