#include "utils/containers/binary_cartesian_product.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_cartesian_product") {
    SUBCASE("both lhs and rhs are nonempty") {
      std::unordered_set<int> lhs = {1, 3};
      std::unordered_set<std::string> rhs = {"a", "b", "c"};

      std::unordered_set<std::pair<int, std::string>> result =
          binary_cartesian_product(lhs, rhs);

      std::unordered_set<std::pair<int, std::string>> correct = {
          {1, "a"},
          {3, "b"},
          {1, "c"},
          {3, "a"},
          {1, "b"},
          {3, "c"},
      };

      CHECK(result == correct);
    }

    SUBCASE("lhs is empty") {
      std::unordered_set<int> lhs = {};
      std::unordered_set<std::string> rhs = {"a", "b", "c"};

      std::unordered_set<std::pair<int, std::string>> result =
          binary_cartesian_product(lhs, rhs);

      std::unordered_set<std::pair<int, std::string>> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("rhs is empty") {
      std::unordered_set<int> lhs = {1, 3};
      std::unordered_set<std::string> rhs = {};

      std::unordered_set<std::pair<int, std::string>> result =
          binary_cartesian_product(lhs, rhs);

      std::unordered_set<std::pair<int, std::string>> correct = {};

      CHECK(result == correct);
    }
  }
}
