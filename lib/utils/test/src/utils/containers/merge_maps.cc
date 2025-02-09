#include "utils/containers/merge_maps.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("merge_disjoint_maps") {
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
          merge_disjoint_maps(l_map, r_map);

      CHECK(result == correct);
    }

    SUBCASE("maps are not disjoint") {
      CHECK_THROWS(merge_disjoint_maps(l_map, l_map));
    }
  }

  TEST_CASE("merge_map_left_dominates") {
    std::unordered_map<int, std::string> l_map = {
        {1, "one"},
        {2, "left_two"},
    };

    std::unordered_map<int, std::string> r_map = {
        {2, "right_two"},
        {3, "three"},
    };

    std::unordered_map<int, std::string> correct = {
        {1, "one"},
        {2, "left_two"},
        {3, "three"},
    };

    std::unordered_map<int, std::string> result =
        merge_map_left_dominates(l_map, r_map);

    CHECK(result == correct);
  }

  TEST_CASE("merge_map_right_dominates") {
    std::unordered_map<int, std::string> l_map = {
        {1, "one"},
        {2, "left_two"},
    };

    std::unordered_map<int, std::string> r_map = {
        {2, "right_two"},
        {3, "three"},
    };

    std::unordered_map<int, std::string> correct = {
        {1, "one"},
        {2, "right_two"},
        {3, "three"},
    };

    std::unordered_map<int, std::string> result =
        merge_map_right_dominates(l_map, r_map);

    CHECK(result == correct);
  }
}
