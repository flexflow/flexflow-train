#include "utils/containers/binary_merge_disjoint_maps.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_merge_disjoint_maps") {
    std::unordered_map<int, std::string> l_map = {
        {1, "one"},
        {2, "two"},
    };

    std::unordered_map<int, std::string> r_map = {
        {3, "three"},
    };

    std::unordered_map<int, std::string> correct = {
        {1, "one"},
        {2, "two"},
        {3, "three"},
    };
    SUBCASE("maps are disjoint") {
      std::unordered_map<int, std::string> result =
          binary_merge_disjoint_maps(l_map, r_map);

      CHECK(result == correct);
    }

    SUBCASE("maps are not disjoint") {
      CHECK_THROWS(binary_merge_disjoint_maps(l_map, l_map));
    }
  }
}
