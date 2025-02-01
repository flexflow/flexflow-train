#include "utils/nonnegative_int/nonnegative_range.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("nonnegative_range(nonnegative_int)") {
    SUBCASE("bound is greater than zero") {
      std::vector<nonnegative_int> result =
          nonnegative_range(nonnegative_int{3});
      std::vector<nonnegative_int> correct = {
          nonnegative_int{0},
          nonnegative_int{1},
          nonnegative_int{2},
      };

      CHECK(result == correct);
    }

    SUBCASE("bound is zero") {
      std::vector<nonnegative_int> result =
          nonnegative_range(nonnegative_int{0});
      std::vector<nonnegative_int> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("nonnegative_range(nonnegative_int, nonnegative_int, int)") {
    std::vector<nonnegative_int> result = nonnegative_range(
        /*start=*/nonnegative_int{7},
        /*end=*/nonnegative_int{3},
        /*step=*/-2);
    std::vector<nonnegative_int> correct = {
        nonnegative_int{7},
        nonnegative_int{5},
    };

    CHECK(result == correct);
  }
}
