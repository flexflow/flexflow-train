#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_ordering.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flatten_dim_coord") {
    DimCoord<int> coord = DimCoord<int>{{
        {7, 0_n},
        {3, 4_n},
        {1, 1_n},
    }};

    DimDomain<int> domain = DimDomain<int>{{
        {7, 2_p},
        {3, 5_p},
        {1, 3_p},
    }};

    DimOrdering<int> dim_ordering = make_dim_ordering_from_vector<int>({
        3,
        7,
        1,
    });

    nonnegative_int result = flatten_dim_coord(coord, domain, dim_ordering);
    nonnegative_int correct = nonnegative_int{4 * 2 * 3 + 0 * 3 + 1};

    CHECK(result == correct);
  }

  TEST_CASE("unflatten_dim_coord") {
    DimDomain<int> domain = DimDomain<int>{{
        {7, 2_p},
        {3, 5_p},
        {1, 3_p},
    }};
    nonnegative_int flattened = nonnegative_int{4 * 2 * 3 + 0 * 3 + 1};

    DimOrdering<int> dim_ordering = make_dim_ordering_from_vector<int>({
        3,
        7,
        1,
    });

    DimCoord<int> result = unflatten_dim_coord(flattened, domain, dim_ordering);
    DimCoord<int> correct = DimCoord<int>{{
        {7, 0_n},
        {3, 4_n},
        {1, 1_n},
    }};

    CHECK(result == correct);
  }
}
