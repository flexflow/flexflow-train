#include "utils/orthotope/eq_projection.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compute_eq_projection") {
    DimCoord<int> coord = DimCoord<int>{{
        {3, 7_n},
        {1, 4_n},
    }};

    EqProjection<int, std::string> projection = EqProjection{
        bidict<int, std::string>{
            {3, "a"},
            {1, "b"},
        },
    };

    DimCoord<std::string> result = compute_eq_projection(projection, coord);
    DimCoord<std::string> correct = DimCoord<std::string>{{
        {"a", 7_n},
        {"b", 4_n},
    }};

    CHECK(result == correct);
  }
}
