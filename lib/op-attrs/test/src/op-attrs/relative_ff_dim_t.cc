#include "op-attrs/relative_ff_dim_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ff_dim_t_from_relative_ff_dim_t_ff_dim_t") {
    int input_dim = 5;

    SUBCASE("Zero") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{0};
      ff_dim_t ff_dim =
          ff_dim_t_from_relative_ff_dim_t_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{nonnegative_int{0}});
    }

    SUBCASE("Positive") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{1};
      ff_dim_t ff_dim =
          ff_dim_t_from_relative_ff_dim_t_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{nonnegative_int{1}});
    }

    SUBCASE("Negative") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{-1};
      ff_dim_t ff_dim =
          ff_dim_t_from_relative_ff_dim_t_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{nonnegative_int{4}});
    }

    SUBCASE("Negative out of range") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{-10};
      CHECK_THROWS(
          ff_dim_t_from_relative_ff_dim_t_ff_dim_t(relative_ff_dim, input_dim));
    }

    SUBCASE("Positive out of range") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{10};
      ff_dim_t ff_dim =
          ff_dim_t_from_relative_ff_dim_t_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{nonnegative_int{10}});
    }
  }
}
