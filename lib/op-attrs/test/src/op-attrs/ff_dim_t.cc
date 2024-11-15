#include "op-attrs/ff_dim_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ff_dim_to_to_relative_ff_dim_t") {
    SUBCASE("Zero") {
      ff_dim_t ff_dim = ff_dim_t{nonnegative_int{0}};
      relative_ff_dim_t relative_ff_dim =
          relative_ff_dim_t_from_ff_dim_t(ff_dim);
      CHECK(relative_ff_dim == relative_ff_dim_t{0});
    }

    SUBCASE("Positive") {
      ff_dim_t ff_dim = ff_dim_t{nonnegative_int{1}};
      relative_ff_dim_t relative_ff_dim =
          relative_ff_dim_t_from_ff_dim_t(ff_dim);
      CHECK(relative_ff_dim == relative_ff_dim_t{1});
    }
  }
}
