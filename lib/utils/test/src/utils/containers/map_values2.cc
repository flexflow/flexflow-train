#include "utils/containers/map_values2.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_values2") {
    std::unordered_map<int, std::string> m = {
        {1, "aa"},
        {2, "aaaaa"},
        {4, "bbb"},
    };

    auto f = [](int k, std::string const &v) -> int { return k + v.size(); };

    std::unordered_map<int, int> result = map_values2(m, f);

    std::unordered_map<int, int> correct = {
        {1, 3},
        {2, 7},
        {4, 7},
    };

    CHECK(result == correct);
  }
}
