#include <doctest/doctest.h>
#include "utils/containers/require_only_key.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("require_only_key") {
    SUBCASE("input has one key that matches") {
      std::unordered_map<int, std::string> m = { {2, "a"} };
      
      std::string result = require_only_key(m, 2);
      std::string correct = "a";

      CHECK(result == correct);
    }

    SUBCASE("input has one key that does not match") {
      std::unordered_map<int, std::string> m = { {3, "a"} };
      
      CHECK_THROWS(require_only_key(m, 2));
    }

    SUBCASE("input is empty") {
      std::unordered_map<int, std::string> m = {};
      
      CHECK_THROWS(require_only_key(m, 2));
    }

    SUBCASE("input has more than one key") {
      std::unordered_map<int, std::string> m = {
        {2, "a"},
        {3, "b"},
      };
      
      CHECK_THROWS(require_only_key(m, 2));
    }
  }
}
