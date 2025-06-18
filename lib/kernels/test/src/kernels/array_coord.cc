#include "kernels/array_coord.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("array_coord_drop_dims") {
    ArrayCoord coord = ArrayCoord{
        FFOrdered{3_n, 5_n, 0_n, 1_n},
    };

    SUBCASE("removes dims specified to be dropped") {
      std::function<bool(ff_dim_t)> should_drop_dim = [](ff_dim_t d) {
        return d.value % 2_n == 0_n;
      };

      ArrayCoord result = array_coord_drop_dims(coord, should_drop_dim);
      ArrayCoord correct = ArrayCoord{
          FFOrdered{5_n, 1_n},
      };

      CHECK(result == correct);
    }

    SUBCASE(
        "is identity function if no dimensions are specified to be dropped") {
      std::function<bool(ff_dim_t)> should_drop_dim = [](ff_dim_t d) {
        return false;
      };

      ArrayCoord result = array_coord_drop_dims(coord, should_drop_dim);
      ArrayCoord correct = coord;

      CHECK(result == correct);
    }

    SUBCASE(
        "returns empty coord if all dimensions are specified to be dropped") {
      std::function<bool(ff_dim_t)> should_drop_dim = [](ff_dim_t d) {
        return true;
      };

      ArrayCoord result = array_coord_drop_dims(coord, should_drop_dim);
      ArrayCoord correct = ArrayCoord{FFOrdered<nonnegative_int>{}};

      CHECK(result == correct);
    }
  }
}
