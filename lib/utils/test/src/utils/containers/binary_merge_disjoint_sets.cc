#include "utils/containers/binary_merge_disjoint_sets.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_merge_disjoint_sets") {
    SUBCASE("sets are disjoint") {
      std::set<int> lhs = {1, 3};
      std::set<int> rhs = {5, 2, 8};

      std::set<int> result = binary_merge_disjoint_sets(lhs, rhs);
      std::set<int> correct = {1, 2, 3, 5, 8};

      CHECK(result == correct);
    }

    SUBCASE("sets are overlapping") {
      std::set<int> lhs = {1, 3};
      std::set<int> rhs = {5, 2, 8, 3};

      CHECK_THROWS(binary_merge_disjoint_sets(lhs, rhs));
    }

    SUBCASE("one set is a subset of the other") {
      std::set<int> lhs = {1, 3};
      std::set<int> rhs = {5, 2, 8, 3, 1};

      CHECK_THROWS(binary_merge_disjoint_sets(lhs, rhs));
    }

    SUBCASE("sets are the same") {
      std::set<int> lhs = {1, 3};
      std::set<int> rhs = lhs;

      CHECK_THROWS(binary_merge_disjoint_sets(lhs, rhs));
    }
  }
}
