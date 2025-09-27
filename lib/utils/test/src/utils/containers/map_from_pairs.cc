#include <doctest/doctest.h>
#include <string>
#include "utils/containers/map_from_pairs.h"
#include "utils/hash/pair.h"
#include "test/utils/doctest/fmt/unordered_map.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_from_pairs") {
    std::unordered_set<std::pair<int, std::string>> input 
      = std::unordered_set<std::pair<int, std::string>>{
      {1, "one"},
      {2, "two"},
    };

    std::unordered_map<int, std::string> result = map_from_pairs(input);

    std::unordered_map<int, std::string> correct = 
      std::unordered_map<int, std::string>{

      {1, "one"},
      {2, "two"},
    };

    CHECK(result == correct);
  }
}
