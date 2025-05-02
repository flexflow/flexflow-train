#include "kernels/array_shape.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_array_coord_set") {
    SUBCASE("ArrayShape is not empty") {
      ArrayShape input = ArrayShape{
          LegionOrdered{2_n, 1_n, 3_n},
      };

      std::unordered_set<ArrayCoord> result = get_array_coord_set(input);
      std::unordered_set<ArrayCoord> correct = {
          ArrayCoord{FFOrdered{0_n, 0_n, 0_n}},
          ArrayCoord{FFOrdered{0_n, 0_n, 1_n}},
          ArrayCoord{FFOrdered{1_n, 0_n, 0_n}},
          ArrayCoord{FFOrdered{1_n, 0_n, 1_n}},
          ArrayCoord{FFOrdered{2_n, 0_n, 0_n}},
          ArrayCoord{FFOrdered{2_n, 0_n, 1_n}},
      };

      CHECK(result == correct);
    }

    SUBCASE("ArrayShape has a dimension of size zero") {
      ArrayShape input = ArrayShape{
          LegionOrdered{2_n, 0_n, 3_n},
      };

      std::unordered_set<ArrayCoord> result = get_array_coord_set(input);
      std::unordered_set<ArrayCoord> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("ArrayShape is zero-dimensional") {
      ArrayShape input = ArrayShape{LegionOrdered<nonnegative_int>{}};

      std::unordered_set<ArrayCoord> result = get_array_coord_set(input);
      std::unordered_set<ArrayCoord> correct = {
          ArrayCoord{FFOrdered<nonnegative_int>{}},
      };

      CHECK(result == correct);
    }
  }
}
