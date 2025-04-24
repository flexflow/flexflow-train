#include "doctest/doctest.h"
#include "kernels/legion_dim.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test LegionDim") {
    SUBCASE("Test add_to_legion_dim") {
      legion_dim_t dim{1_n};
      CHECK(add_to_legion_dim(dim, 2) == legion_dim_t{3_n});
    }

    SUBCASE("Test legion_dim_from_ff_dim") {
      CHECK(legion_dim_from_ff_dim(ff_dim_t{0_n}, 4_n) == legion_dim_t{3_n});
      CHECK(legion_dim_from_ff_dim(ff_dim_t{1_n}, 4_n) == legion_dim_t{2_n});
      CHECK(legion_dim_from_ff_dim(ff_dim_t{2_n}, 4_n) == legion_dim_t{1_n});
      CHECK(legion_dim_from_ff_dim(ff_dim_t{3_n}, 4_n) == legion_dim_t{0_n});
    }

    SUBCASE("Test LegionOrdered") {
      LegionOrdered<int> legion_ordered{1, 2, 3, 4};

      SUBCASE("Test ff_ordered_from_legion_ordered") {
        CHECK(ff_ordered_from_legion_ordered(legion_ordered) ==
              FFOrdered<int>{4, 3, 2, 1});
      }
    }
  }
}
