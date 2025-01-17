#include "utils/containers/scanr1.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <string>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("scanr1") {
    SUBCASE("sum") {
      std::vector<int> input = {1, 2, 3, 4};
      std::vector<int> result =
          scanr1(input, [](int x, int accum) { return x + accum; });
      std::vector<int> correct = {10, 9, 7, 4};
      CHECK(result == correct);
    }

    SUBCASE("noncommutative function") {
      std::vector<int> input = {1, 2, 5, 2};
      auto f = [](int x, int accum) { return accum - x; };
      std::vector<int> result = scanr1(input, f);
      std::vector<int> correct = {-6, -5, -3, 2};
      CHECK(result == correct);
    }

    SUBCASE("empty input") {
      std::vector<int> input = {};
      std::vector<int> result =
          scanr1(input, [](int x, int accum) -> int { throw std::runtime_error("should not be called"); });
      std::vector<int> correct = {};
      CHECK(result == correct);
    }
  }
}
