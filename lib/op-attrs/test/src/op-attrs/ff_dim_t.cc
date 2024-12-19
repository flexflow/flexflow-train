#include "op-attrs/ff_dim_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("FF_DIM_T_TO_RELATIVE_FF_DIM_T") {
    SUBCASE("ZERO") {
      ff_dim_t ff_dim = ff_dim_t{nonnegative_int{0}};
      relative_ff_dim_t relative_ff_dim = ff_dim_t_to_relative_ff_dim_t(ff_dim);
      CHECK(relative_ff_dim == relative_ff_dim_t{0});
    }

    SUBCASE("POSITIVE") {
      ff_dim_t ff_dim = ff_dim_t{nonnegative_int{1}};
      relative_ff_dim_t relative_ff_dim = ff_dim_t_to_relative_ff_dim_t(ff_dim);
      CHECK(relative_ff_dim == relative_ff_dim_t{1});
    }
  }
}
