#include "utils/orthotope/dim_coord.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flatten_coord") {
    DimCoord<int> coord = DimCoord<int>{{
      {3, 4_n},
      {7, 0_n},
      {1, 1_n},
    }};

    DimDomain<int> domain = DimDomain<int>{{
      {3, 5_p},
      {7, 2_p},
      {1, 3_p},
    }};

    nonnegative_int result = flatten_dim_coord(coord, domain);
    nonnegative_int correct = nonnegative_int{1 * 2 * 5 + 4 * 2 + 0};

    CHECK(result == correct);
  }

  TEST_CASE("unflatten_dim_coord") {
    NOT_IMPLEMENTED(); 
  }
}
