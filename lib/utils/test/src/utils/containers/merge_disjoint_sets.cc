#include "utils/containers/merge_disjoint_sets.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("merge_disjoint_sets") {
    SUBCASE("input is empty") {
      std::vector<std::set<int>> input = {};

      std::set<int> result = merge_disjoint_sets(input);
      std::set<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("sets are disjoint") {
      std::vector<std::set<int>> input = {
          {1, 3},
          {5, 2, 8},
          {-1},
          {9, 10, 11, 12},
      };

      std::set<int> result = merge_disjoint_sets(input);
      std::set<int> correct = {
          1,
          3,
          5,
          2,
          8,
          -1,
          9,
          10,
          11,
          12,
      };

      CHECK(result == correct);
    }

    SUBCASE("sets are not disjoint") {
      std::vector<std::set<int>> input = {
          {1, 3},
          {5, 2, 8},
          {-1, 1},
          {9, 10, 11, 12},
      };

      CHECK_THROWS(merge_disjoint_sets(input));
    }

    SUBCASE("input contains duplicate sets") {
      std::vector<std::set<int>> input = {
          {1, 3},
          {1, 3},
          {1, 3},
      };

      CHECK_THROWS(merge_disjoint_sets(input));
    }
  }
}
