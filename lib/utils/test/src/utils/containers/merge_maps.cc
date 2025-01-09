#include "utils/containers/merge_maps.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("merge_maps") {
    SUBCASE("maps are disjoint") {
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

      SUBCASE("MergeMethod::REQUIRE_DISJOINT") {
        std::unordered_map<int, std::string> result =
            merge_maps(l_map, r_map, MergeMethod::REQUIRE_DISJOINT);

        CHECK(result == correct);
      }

      SUBCASE("MergeMethod::LEFT_DOMINATES") {
        std::unordered_map<int, std::string> result =
            merge_maps(l_map, r_map, MergeMethod::LEFT_DOMINATES);

        CHECK(result == correct);
      }

      SUBCASE("MergeMethod::RIGHT_DOMINATES") {
        std::unordered_map<int, std::string> result =
            merge_maps(l_map, r_map, MergeMethod::RIGHT_DOMINATES);

        CHECK(result == correct);
      }
    }

    SUBCASE("maps are not disjoint") {
      std::unordered_map<int, std::string> l_map = {
          {1, "one"},
          {2, "left_two"},
      };

      std::unordered_map<int, std::string> r_map = {
          {2, "right_two"},
          {3, "three"},
      };

      SUBCASE("MergeMethod::REQUIRE_DISJOINT") {
        CHECK_THROWS(merge_maps(l_map, r_map, MergeMethod::REQUIRE_DISJOINT));
      }

      SUBCASE("MergeMethod::LEFT_DOMINATES") {
        std::unordered_map<int, std::string> correct = {
            {1, "one"},
            {2, "left_two"},
            {3, "three"},
        };

        std::unordered_map<int, std::string> result =
            merge_maps(l_map, r_map, MergeMethod::LEFT_DOMINATES);

        CHECK(result == correct);
      }

      SUBCASE("MergeMethod::RIGHT_DOMINATES") {
        std::unordered_map<int, std::string> correct = {
            {1, "one"},
            {2, "right_two"},
            {3, "three"},
        };

        std::unordered_map<int, std::string> result =
            merge_maps(l_map, r_map, MergeMethod::RIGHT_DOMINATES);

        CHECK(result == correct);
      }
    }
  }
}
