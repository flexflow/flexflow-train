#include "utils/containers/get_last.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_last") {
    SUBCASE("vector is empty") {
      std::vector<int> v = {};

      CHECK_THROWS(get_last(v));
    }

    SUBCASE("vector is not empty") {
      std::vector<int> v = {1, 3, 2};

      int result = get_last(v);
      int correct = 2;

      CHECK(result == correct);
    }
  }
}
