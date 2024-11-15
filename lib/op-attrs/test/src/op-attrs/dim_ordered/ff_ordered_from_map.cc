#include "op-attrs/dim_ordered/ff_ordered_from_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("ff_ordered_from_map",
                     T,
                     std::map<ff_dim_t, int>,
                     std::unordered_map<ff_dim_t, int>) {
    SUBCASE("input is empty") {
      T m = {};

      FFOrdered<int> result = ff_ordered_from_map(m);
      FFOrdered<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("input is missing keys") {
      SUBCASE("missing key is in middle") {
        T m = {
            {ff_dim_t{nonnegative_int{0}}, 4},
            {ff_dim_t{nonnegative_int{1}}, 2},
            {ff_dim_t{nonnegative_int{3}}, 5},
        };

        CHECK_THROWS(ff_ordered_from_map(m));
      }

      SUBCASE("missing key is 0 idx") {
        T m = {
            {ff_dim_t{nonnegative_int{1}}, 2},
            {ff_dim_t{nonnegative_int{2}}, 7},
            {ff_dim_t{nonnegative_int{3}}, 5},
        };

        CHECK_THROWS(ff_ordered_from_map(m));
      }
    }

    SUBCASE("input is valid") {
      T m = {
          {ff_dim_t{nonnegative_int{0}}, 4},
          {ff_dim_t{nonnegative_int{1}}, 5},
          {ff_dim_t{nonnegative_int{2}}, 2},
          {ff_dim_t{nonnegative_int{3}}, 7},
      };

      FFOrdered<int> result = ff_ordered_from_map(m);
      FFOrdered<int> correct = {4, 5, 2, 7};

      CHECK(result == correct);
    }
  }
}
