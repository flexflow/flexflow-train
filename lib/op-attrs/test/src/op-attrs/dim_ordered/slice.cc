#include "op-attrs/dim_ordered/slice.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("slice(FFOrdered<T>, ..., ...") {
    FFOrdered<size_t> d = FFOrdered<size_t>{
        1,
        2,
        3,
        4,
    };
    SUBCASE("ff_dim_t, ff_dim_t") {
      FFOrdered<size_t> result =
          slice(d, ff_dim_t{nonnegative_int{1}}, ff_dim_t{nonnegative_int{3}});
      FFOrdered<size_t> correct = FFOrdered<size_t>{2, 3};

      CHECK(result == correct);
    }
    SUBCASE("ff_dim_t, std::nullopt_t") {
      FFOrdered<size_t> result =
          slice(d, ff_dim_t{nonnegative_int{1}}, std::nullopt);
      FFOrdered<size_t> correct = FFOrdered<size_t>{2, 3, 4};

      CHECK(result == correct);
    }
    SUBCASE("std::nullopt_t, ff_dim_t") {
      FFOrdered<size_t> result =
          slice(d, std::nullopt, ff_dim_t{nonnegative_int{3}});
      FFOrdered<size_t> correct = FFOrdered<size_t>{1, 2, 3};

      CHECK(result == correct);
    }
    SUBCASE("relative_ff_dim_t, relative_ff_dim_t") {
      FFOrdered<size_t> result =
          slice(d, relative_ff_dim_t{1}, relative_ff_dim_t{-1});
      FFOrdered<size_t> correct = FFOrdered<size_t>{2, 3};

      CHECK(result == correct);
    }
    SUBCASE("relative_ff_dim_t, std::nullopt_t") {
      FFOrdered<size_t> result = slice(d, relative_ff_dim_t{-3}, std::nullopt);
      FFOrdered<size_t> correct = FFOrdered<size_t>{2, 3, 4};

      CHECK(result == correct);
    }
    SUBCASE("std::nullopt_t, relative_ff_dim_t") {
      FFOrdered<size_t> result = slice(d, std::nullopt, relative_ff_dim_t{-1});
      FFOrdered<size_t> correct = FFOrdered<size_t>{1, 2, 3};

      CHECK(result == correct);
    }
    SUBCASE("start index = stop index") {
      FFOrdered<size_t> result =
          slice(d, relative_ff_dim_t{1}, relative_ff_dim_t{1});
      FFOrdered<size_t> correct = FFOrdered<size_t>{};

      CHECK(result == correct);
    }
    SUBCASE("start index = stop index (using negative indexing)") {
      FFOrdered<size_t> result =
          slice(d, relative_ff_dim_t{1}, relative_ff_dim_t{-3});
      FFOrdered<size_t> correct = FFOrdered<size_t>{};

      CHECK(result == correct);
    }
    SUBCASE("start index > stop index") {
      FFOrdered<size_t> result =
          slice(d, relative_ff_dim_t{1}, relative_ff_dim_t{0});
      FFOrdered<size_t> correct = FFOrdered<size_t>{};

      CHECK(result == correct);
    }
    SUBCASE("start index > stop index (using negative indexing)") {
      FFOrdered<size_t> result =
          slice(d, relative_ff_dim_t{1}, relative_ff_dim_t{-4});
      FFOrdered<size_t> correct = FFOrdered<size_t>{};

      CHECK(result == correct);
    }
    SUBCASE("start index out of bounds (too low)") {
      CHECK_THROWS(slice(d, relative_ff_dim_t{-10}, std::nullopt));
    }
    SUBCASE("start index out of bounds (too high)") {
      CHECK_THROWS(slice(d, relative_ff_dim_t{10}, std::nullopt));
    }
    SUBCASE("stop index out of bounds (too low)") {
      CHECK_THROWS(slice(d, std::nullopt, relative_ff_dim_t{-10}));
    }
    SUBCASE("stop index out of bounds (too high)") {
      CHECK_THROWS(slice(d, std::nullopt, relative_ff_dim_t{10}));
    }
  }
}
