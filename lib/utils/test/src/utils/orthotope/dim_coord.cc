#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_ordering.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("lift_dim_coord") {
    DimCoord<int> coord = DimCoord<int>{{
        {7, 0_n},
        {3, 4_n},
        {1, 1_n},
    }};

    SUBCASE("lifted dims are a superset of coord dims") {
      std::unordered_set<int> lifted_dims = {1, 3, 6, 7};

      DimCoord<int> result = lift_dim_coord(coord, lifted_dims);
      DimCoord<int> correct = DimCoord<int>{{
        {7, 0_n},
        {3, 4_n},
        {1, 1_n},
        {6, 0_n},
      }};

      CHECK(result == correct);
    }

    SUBCASE("lifted dims are the same as coord dims") {
      std::unordered_set<int> lifted_dims = {1, 3, 7};

      DimCoord<int> result = lift_dim_coord(coord, lifted_dims);
      DimCoord<int> correct = coord;

      CHECK(result == correct);
    }

    SUBCASE("lifted dims are a subset of coord dims") {
      std::unordered_set<int> lifted_dims = {1, 7};

      CHECK_THROWS(lift_dim_coord(coord, lifted_dims));
    }

    SUBCASE("lifted dims are overlapping with coord dims") {
      std::unordered_set<int> lifted_dims = {1, 2, 7};

      CHECK_THROWS(lift_dim_coord(coord, lifted_dims));
    }
  }

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
