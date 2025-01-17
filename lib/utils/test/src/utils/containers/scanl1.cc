#include "utils/containers/scanl1.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <string>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("scanl1") {
    SUBCASE("sum") {
      std::vector<int> input = {1, 2, 3, 4};
      std::vector<int> result =
          scanl1(input, [](int accum, int x) { return accum + x; });
      std::vector<int> correct = {1, 3, 6, 10};
      CHECK(result == correct);
    }

    SUBCASE("noncommutative function") {
      std::vector<int> input = {1, 2, 5, 2};
      auto f = [](int accum, int x) { return accum - x; };
      std::vector<int> result = scanl1(input, f);
      std::vector<int> correct = {1, -1, -6, -8};
      CHECK(result == correct);
    }

    SUBCASE("empty input") {
      std::vector<int> input = {};
      std::vector<int> result =
          scanl1(input, [](int x, int accum) -> int { throw std::runtime_error("should not be called"); });
      std::vector<int> correct = {};
      CHECK(result == correct);
    }
  }
}
