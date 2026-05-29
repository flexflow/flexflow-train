#include "utils/containers/map_from_pairs.h"
#include "test/utils/doctest/fmt/map.h"
#include <doctest/doctest.h>
#include <string>
#include <set>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_from_pairs") {
    std::set<std::pair<int, std::string>> input =
        std::set<std::pair<int, std::string>>{
            {1, "one"},
            {2, "two"},
        };

    std::map<int, std::string> result = map_from_pairs(input);

    std::map<int, std::string> correct =
        std::map<int, std::string>{
            {1, "one"},
            {2, "two"},
        };

    CHECK(result == correct);
  }
}
