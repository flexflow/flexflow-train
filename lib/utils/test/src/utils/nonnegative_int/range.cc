#include "utils/nonnegative_int/range.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("range(nonnegative_int, nonnegative_int, int)") {
    SUBCASE("step = 1") {
      nonnegative_int start = nonnegative_int{3};
      nonnegative_int end = nonnegative_int{5};

      std::vector<nonnegative_int> result = range(start, end);
      std::vector<nonnegative_int> correct = {
          nonnegative_int{3},
          nonnegative_int{4},
      };

      CHECK(result == correct);
    }

    SUBCASE("step = -1") {
      nonnegative_int start = nonnegative_int{7};
      nonnegative_int end = nonnegative_int{4};

      std::vector<nonnegative_int> result = range(start, end, -1);
      std::vector<nonnegative_int> correct = {
          nonnegative_int{7},
          nonnegative_int{6},
          nonnegative_int{5},
      };

      CHECK(result == correct);
    }

    SUBCASE("step = 0") {
      SUBCASE("output is nonempty") {
        CHECK_THROWS(range(nonnegative_int{2}, nonnegative_int{5}, 0));
      }

      SUBCASE("output is empty") {
        CHECK_THROWS(range(nonnegative_int{2}, nonnegative_int{2}, 0));
      }
    }
  }

  TEST_CASE("range(nonnegative_int)") {
    SUBCASE("end is zero") {
      nonnegative_int end = nonnegative_int{0};

      std::vector<nonnegative_int> result = range(end);
      std::vector<nonnegative_int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("end is nonzero") {
      nonnegative_int end = nonnegative_int{3};

      std::vector<nonnegative_int> result = range(end);
      std::vector<nonnegative_int> correct = {
          nonnegative_int{0},
          nonnegative_int{1},
          nonnegative_int{2},
      };

      CHECK(result == correct);
    }
  }
}
