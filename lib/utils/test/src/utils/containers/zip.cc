#include <doctest/doctest.h>
#include "utils/containers/zip.h"
#include <string>
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/doctest/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip(std::vector<L>, std::vector<R>)") {
    SUBCASE("L and R types are the same") {
      std::vector<int> lhs = {2, 1, 2};
      std::vector<int> rhs = {5, 4, 8};

      std::vector<std::pair<int, int>> result = zip(lhs, rhs);
      std::vector<std::pair<int, int>> correct = {{2, 5}, {1, 4}, {2, 8}};

      CHECK(result == correct);
    }

    SUBCASE("L and R types are different") {
      std::vector<std::string> lhs = {"a", "b", "b"};
      std::vector<int> rhs = {5, 4, 8};

      std::vector<std::pair<std::string, int>> result = zip(lhs, rhs);
      std::vector<std::pair<std::string, int>> correct = {{"a", 5}, {"b", 4}, {"b", 8}};

      CHECK(result == correct);
    }

    SUBCASE("left is longer than right") {
      std::vector<int> lhs = {2, 1, 2};
      std::vector<int> rhs = {5, 4};

      std::vector<std::pair<int, int>> result = zip(lhs, rhs);
      std::vector<std::pair<int, int>> correct = {{2, 5}, {1, 4}};

      CHECK(result == correct);
    }

    SUBCASE("right is longer than left") {
      std::vector<int> lhs = {2};
      std::vector<int> rhs = {5, 4, 8};

      std::vector<std::pair<int, int>> result = zip(lhs, rhs);
      std::vector<std::pair<int, int>> correct = {{2, 5}};

      CHECK(result == correct);
    }

    SUBCASE("left is empty") {
      std::vector<int> lhs = {};
      std::vector<int> rhs = {5, 4, 8};

      std::vector<std::pair<int, int>> result = zip(lhs, rhs);
      std::vector<std::pair<int, int>> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("right is empty") {
      std::vector<int> lhs = {2, 1, 2};
      std::vector<int> rhs = {};

      std::vector<std::pair<int, int>> result = zip(lhs, rhs);
      std::vector<std::pair<int, int>> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("both are empty") {
      std::vector<int> lhs = {};
      std::vector<int> rhs = {};

      std::vector<std::pair<int, int>> result = zip(lhs, rhs);
      std::vector<std::pair<int, int>> correct = {};

      CHECK(result == correct);
    }
  }
}
