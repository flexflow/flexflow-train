#include <doctest/doctest.h>
#include <string>
#include "utils/containers/unstructured_exhaustive_relational_join.h"
#include "utils/hash/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unstructured_exhaustive_relational_join") {
    SUBCASE("join is exhaustive") {
      std::unordered_set<std::pair<int, std::string>> lhs = {
        {1, "one"},
        {1, "odd"},
        {2, "two"},
        {3, "three"},
        {3, "odd"},
      };

      std::unordered_set<std::pair<std::string, bool>> rhs = {
        {"one", false},
        {"odd", true},
        {"two", true},
        {"three", true},
      };

      std::unordered_set<std::pair<int, bool>> result = unstructured_exhaustive_relational_join(lhs, rhs);
      std::unordered_set<std::pair<int, bool>> correct = {
        {1, false},
        {1, true},
        {2, true},
        {3, true},
      };

      CHECK(result == correct);
    }

    SUBCASE("join is not exhaustive in lhs") {
      std::unordered_set<std::pair<int, std::string>> lhs = {
        {1, "one"},
        {1, "odd"},
        {2, "two"},
        {3, "odd"},
      };

      std::unordered_set<std::pair<std::string, bool>> rhs = {
        {"one", false},
        {"odd", true},
        {"two", true},
        {"three", true},
      };

      CHECK_THROWS(unstructured_exhaustive_relational_join(lhs, rhs));
    }

    SUBCASE("join is not exhaustive in rhs") {
      std::unordered_set<std::pair<int, std::string>> lhs = {
        {1, "one"},
        {1, "odd"},
        {2, "two"},
        {3, "three"},
        {3, "odd"},
      };

      std::unordered_set<std::pair<std::string, bool>> rhs = {
        {"one", false},
        {"odd", true},
        {"two", true},
      };

      CHECK_THROWS(unstructured_exhaustive_relational_join(lhs, rhs));
    }
  }
}
