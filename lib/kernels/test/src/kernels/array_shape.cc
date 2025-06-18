#include "kernels/array_shape.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_array_coord_set") {
    SUBCASE("ArrayShape is not empty") {
      ArrayShape input = ArrayShape{
          LegionOrdered{2_p, 1_p, 3_p},
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

    SUBCASE("ArrayShape is zero-dimensional") {
      ArrayShape input = ArrayShape{LegionOrdered<positive_int>{}};

      std::unordered_set<ArrayCoord> result = get_array_coord_set(input);
      std::unordered_set<ArrayCoord> correct = {
          ArrayCoord{FFOrdered<nonnegative_int>{}},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("array_shape_drop_dims") {
    ArrayShape input = ArrayShape{
        LegionOrdered{2_p, 4_p, 3_p},
    };

    SUBCASE("removes dims specified to be dropped") {
      auto should_drop_dim = [](ff_dim_t dim) -> bool {
        return dim.value % 2_n == 0;
      };

      ArrayShape result = array_shape_drop_dims(input, should_drop_dim);
      ArrayShape correct = ArrayShape{
          LegionOrdered{4_p},
      };

      CHECK(result == correct);
    }

    SUBCASE(
        "is identity function if no dimensions are specified to be dropped") {
      auto should_drop_dim = [](ff_dim_t dim) -> bool { return false; };

      ArrayShape result = array_shape_drop_dims(input, should_drop_dim);
      ArrayShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE(
        "is identity function if no dimensions are specified to be dropped") {
      auto should_drop_dim = [](ff_dim_t dim) -> bool { return false; };

      ArrayShape result = array_shape_drop_dims(input, should_drop_dim);
      ArrayShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE(
        "returns empty shape if all dimensions are specified to be dropped") {
      auto should_drop_dim = [](ff_dim_t dim) -> bool { return true; };

      ArrayShape result = array_shape_drop_dims(input, should_drop_dim);
      ArrayShape correct = ArrayShape{LegionOrdered<positive_int>{}};

      CHECK(result == correct);
    }
  }
}
