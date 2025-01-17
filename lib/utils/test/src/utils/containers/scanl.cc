#include "utils/containers/scanl.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <string>
#include <vector>

using namespace FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("scanl") {
    SUBCASE("sum") {
      std::vector<int> input = {1, 2, 3, 4};
      std::vector<int> result =
          scanl(input, 0, [](int accum, int x) { return accum + x; });
      std::vector<int> correct = {0, 1, 3, 6, 10};
      CHECK(result == correct);
    }

    SUBCASE("noncommutative function") {
      std::vector<int> input = {1, 3, 1, 2};
      auto op = [](int accum, int x) { return accum - x; };
      std::vector<int> result = scanl(input, 1, op);
      std::vector<int> correct = {1, 0, -3, -4, -6};
      CHECK(result == correct);
    }

    SUBCASE("heterogeneous types") {
      std::vector<int> input = {1, 2, 3, 4};
      auto op = [](std::string const &a, int b) {
        return a + std::to_string(b);
      };
      std::vector<std::string> result = scanl(input, std::string(""), op);
      std::vector<std::string> correct = {"", "1", "12", "123", "1234"};
      CHECK(result == correct);
    }

    SUBCASE("empty input") {
      std::vector<int> input = {};
      std::vector<int> result =
          scanl(input, 2, [](int accum, int x) -> int { throw std::runtime_error("should not be called"); });
      std::vector<int> correct = {2};
      CHECK(result == correct);
    }
  }
}
