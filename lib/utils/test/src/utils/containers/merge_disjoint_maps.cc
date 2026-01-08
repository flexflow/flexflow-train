#include "utils/containers/merge_disjoint_maps.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("merge_disjoint_maps") {
    std::unordered_map<int, std::string> m1 = {
        {4, "four"},
        {2, "two"},
    };

    std::unordered_map<int, std::string> m2 = {
        {3, "four"},
    };

    std::unordered_map<int, std::string> m3 = {
        {1, "one"},
    };

    std::unordered_map<int, std::string> m4 = {};

    SUBCASE("maps are disjoint") {
      std::vector<std::unordered_map<int, std::string>> input = {
          m1,
          m2,
          m3,
          m4,
      };

      std::unordered_map<int, std::string> result = merge_disjoint_maps(input);

      std::unordered_map<int, std::string> correct = {
          {4, "four"},
          {2, "two"},
          {3, "four"},
          {1, "one"},
      };

      CHECK(result == correct);
    }

    SUBCASE("maps are not disjoint") {
      std::unordered_map<int, std::string> m5 = {
          {4, "five"},
          {6, "six"},
      };

      std::vector<std::unordered_map<int, std::string>> input = {
          m1,
          m2,
          m3,
          m4,
          m5,
      };

      CHECK_THROWS(merge_disjoint_maps(input));
    }

    SUBCASE("maps are not disjoint but have identical values") {
      std::unordered_map<int, std::string> m5 = {
          {4, "four"},
          {6, "six"},
      };

      std::vector<std::unordered_map<int, std::string>> input = {
          m1,
          m2,
          m3,
          m4,
          m5,
      };

      CHECK_THROWS(merge_disjoint_maps(input));
    }
  }
}
