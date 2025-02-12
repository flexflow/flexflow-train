#include "utils/containers/contains.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("contains") {
    SUBCASE("std::vector") {
      std::vector<int> v = {1, 2, 3, 4, 5};
      CHECK(contains(v, 3));
      CHECK_FALSE(contains(v, 6));
    }

    SUBCASE("std::unordered_set") {
      std::unordered_set<int> s = {1, 2, 3, 4, 5};
      CHECK(contains(s, 3));
      CHECK_FALSE(contains(s, 6));
    }
  }
}
