#include <doctest/doctest.h>
#include <vector>
#include "utils/containers/all_are_true.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("all_are_true") {
    SUBCASE("all elements are true") {
      std::vector<bool> input = {true, true, true};

      bool result = all_are_true(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("not all elements are true") {
      std::vector<bool> input = {true, false, true, false};      

      bool result = all_are_true(input);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("empty input vector") {
      std::vector<bool> input = {};

      bool result = all_are_true(input);
      bool correct = true;

      CHECK(result == correct);
    }
  }
}
