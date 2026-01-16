#include "utils/containers/at_idx.h"
#include "test/utils/doctest/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("at_idx(std::vector<E>, nonnegative_int)") {
    std::vector<int> vec = {1, 3, 2, 3};

    SUBCASE("idx is in bounds") {
      nonnegative_int idx = 1_n;

      int result = at_idx(vec, idx);
      int correct = 3;

      CHECK(result == correct);
    }

    SUBCASE("idx is out of bounds") {
      nonnegative_int idx = 4_n;

      CHECK_THROWS(at_idx(vec, idx));
    }
  }
}
