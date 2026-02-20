#include "utils/containers/binary_merge_maps_with_right_dominating.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_merge_maps_with_right_dominating") {
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
        binary_merge_maps_with_right_dominating(l_map, r_map);

    CHECK(result == correct);
  }
}
