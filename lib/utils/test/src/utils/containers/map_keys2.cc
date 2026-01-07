#include "utils/containers/map_keys2.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_keys2") {
    SUBCASE("output keys are unique") {
      std::unordered_map<int, std::string> m = {
          {1, "aa"},
          {2, "aaaaa"},
      };

      auto f = [](int k, std::string const &v) -> std::string {
        return std::to_string(k + v.size());
      };

      std::unordered_map<std::string, std::string> result = map_keys2(m, f);

      std::unordered_map<std::string, std::string> correct = {
          {"3", "aa"},
          {"7", "aaaaa"},
      };

      CHECK(result == correct);
    }

    SUBCASE("output keys are non-unique") {
      std::unordered_map<int, std::string> m = {
          {1, "aa"},
          {2, "aaaaa"},
      };

      auto f = [](int k, std::string const &v) -> std::string { return "b"; };

      CHECK_THROWS(map_keys2(m, f));
    }
  }
}
