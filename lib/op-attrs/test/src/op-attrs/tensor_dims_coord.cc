#include "op-attrs/tensor_dims_coord.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_dims_coord_drop_dims") {
    TensorDimsCoord coord = TensorDimsCoord{
        FFOrdered{3_n, 5_n, 0_n, 1_n},
    };

    SUBCASE("removes dims specified to be dropped") {
      std::function<bool(ff_dim_t)> should_drop_dim = [](ff_dim_t d) {
        return d.value % 2_n == 0_n;
      };

      TensorDimsCoord result =
          tensor_dims_coord_drop_dims(coord, should_drop_dim);
      TensorDimsCoord correct = TensorDimsCoord{
          FFOrdered{5_n, 1_n},
      };

      CHECK(result == correct);
    }

    SUBCASE(
        "is identity function if no dimensions are specified to be dropped") {
      std::function<bool(ff_dim_t)> should_drop_dim = [](ff_dim_t d) {
        return false;
      };

      TensorDimsCoord result =
          tensor_dims_coord_drop_dims(coord, should_drop_dim);
      TensorDimsCoord correct = coord;

      CHECK(result == correct);
    }

    SUBCASE(
        "returns empty coord if all dimensions are specified to be dropped") {
      std::function<bool(ff_dim_t)> should_drop_dim = [](ff_dim_t d) {
        return true;
      };

      TensorDimsCoord result =
          tensor_dims_coord_drop_dims(coord, should_drop_dim);
      TensorDimsCoord correct = TensorDimsCoord{FFOrdered<nonnegative_int>{}};

      CHECK(result == correct);
    }
  }
}
