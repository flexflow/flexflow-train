#include "utils/containers/any_are_true.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("any_are_true") {
    SUBCASE("all elements are false") {
      std::vector<bool> input = {false, false, false};

      bool result = any_are_true(input);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("not all elements are false") {
      std::vector<bool> input = {false, true, false, true};

      bool result = any_are_true(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("empty input vector") {
      std::vector<bool> input = {};

      bool result = any_are_true(input);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
