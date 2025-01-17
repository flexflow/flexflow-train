#include "utils/containers/scanr.h"
#include <doctest/doctest.h>
#include <string>
#include "test/utils/doctest/fmt/vector.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("scanr") {
    SUBCASE("sum") {
      std::vector<int> input = {1, 2, 3, 4};
      std::vector<int> result =
          scanr(input, 0, [](int x, int accum) { return x + accum; });
      std::vector<int> correct = {10, 9, 7, 4, 0};
      CHECK(result == correct);
    }

    SUBCASE("noncommutative function") {
      std::vector<int> input = {1, 3, 1, 2};
      auto op = [](int x, int accum) { return accum - x; };
      std::vector<int> result = scanr(input, 1, op);
      std::vector<int> correct = {-6, -5, -2, -1, 1};
      CHECK(result == correct);
    }

    SUBCASE("heterogeneous types") {
      std::vector<int> input = {1, 2, 3, 4};
      auto op = [](int x, std::string const &accum) {
        return accum + std::to_string(x);
      };
      std::vector<std::string> result = scanr(input, std::string(""), op);
      std::vector<std::string> correct = {"4321", "432", "43", "4", ""};
      CHECK(result == correct);
    }

    SUBCASE("empty input") {
      std::vector<int> input = {};
      std::vector<int> result =
          scanr(input, 2, [](int x, int accum) -> int { throw std::runtime_error("should not be called"); });
      std::vector<int> correct = {2};
      CHECK(result == correct);
    }
  }
}
