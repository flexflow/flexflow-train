#include "op-attrs/ff_ordered/zip.h"
#include "test/utils/doctest/fmt/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip(FFOrdered<T1>, FFOrdered<T2>)") {
    FFOrdered<int> lhs_input = {9, 9, 8, 9};
    FFOrdered<std::string> rhs_input = {"m", "m", "k", "l", "m"};

    SUBCASE("lhs is longer") {
      FFOrdered<std::pair<int, std::string>> result = zip(lhs_input, rhs_input);

      FFOrdered<std::pair<int, std::string>> correct = {
          {9, "m"},
          {9, "m"},
          {8, "k"},
          {9, "l"},
      };

      CHECK(result == correct);
    }

    SUBCASE("rhs is longer") {
      FFOrdered<std::pair<std::string, int>> result = zip(rhs_input, lhs_input);

      FFOrdered<std::pair<std::string, int>> correct = {
          {"m", 9},
          {"m", 9},
          {"k", 8},
          {"l", 9},
      };

      CHECK(result == correct);
    }
  }
}
