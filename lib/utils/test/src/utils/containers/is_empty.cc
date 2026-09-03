#include "utils/containers/is_empty.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_empty") {
    SUBCASE("container is empty") {
      std::vector<int> input = {};

      bool result = is_empty(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("container is not empty") {
      std::vector<int> input = {1};

      bool result = is_empty(input);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
