#include "kernels/legion_ordered/slice.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("slice(LegionOrdered<T>, ..., ...") {
    LegionOrdered<size_t> d = LegionOrdered<size_t>{
        1,
        2,
        3,
        4,
    };
    SUBCASE("legion_dim_t, legion_dim_t") {
      LegionOrdered<size_t> result =
          slice(d, legion_dim_t{nonnegative_int{1}}, legion_dim_t{nonnegative_int{3}});
      LegionOrdered<size_t> correct = LegionOrdered<size_t>{2, 3};

      CHECK(result == correct);
    }
    SUBCASE("legion_dim_t, std::nullopt_t") {
      LegionOrdered<size_t> result =
          slice(d, legion_dim_t{nonnegative_int{1}}, std::nullopt);
      LegionOrdered<size_t> correct = LegionOrdered<size_t>{2, 3, 4};

      CHECK(result == correct);
    }
  }
}
