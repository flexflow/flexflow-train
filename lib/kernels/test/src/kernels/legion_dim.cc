#include "kernels/legion_dim.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("key_range(LegionOrdered<T>)") {
    SUBCASE("input is non-empty") {
      LegionOrdered<int> input = {5, 3, 2, 3};

      std::set<legion_dim_t> result = key_range(input);
      std::set<legion_dim_t> correct = {
          legion_dim_t{0_n},
          legion_dim_t{1_n},
          legion_dim_t{2_n},
          legion_dim_t{3_n},
      };

      CHECK(result == correct);
    }

    SUBCASE("input is empty") {
      LegionOrdered<int> input = {};

      std::set<legion_dim_t> result = key_range(input);
      std::set<legion_dim_t> correct = {};

      CHECK(result == correct);
    }
  }
}
