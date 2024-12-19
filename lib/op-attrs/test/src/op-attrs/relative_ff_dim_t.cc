#include "op-attrs/relative_ff_dim_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RELATIVE_FF_DIM_T_TO_FF_DIM_T") {
    int input_dim = 5;

    SUBCASE("ZERO") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{0};
      ff_dim_t ff_dim =
          relative_ff_dim_t_to_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{nonnegative_int{0}});
    }

    SUBCASE("POSITIVE") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{1};
      ff_dim_t ff_dim =
          relative_ff_dim_t_to_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{nonnegative_int{1}});
    }

    SUBCASE("NEGATIVE") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{-1};
      ff_dim_t ff_dim =
          relative_ff_dim_t_to_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{nonnegative_int{4}});
    }

    SUBCASE("OUT OF RANGE") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{-10};
      CHECK_THROWS(relative_ff_dim_t_to_ff_dim_t(relative_ff_dim, input_dim));
    }
  }
}
