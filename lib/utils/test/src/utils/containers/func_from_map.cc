#include "utils/containers/func_from_map.h"
#include <doctest/doctest.h>
#include <string>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("func_from_map") {

    std::unordered_map<std::string, int> map = {{"a", 1}, {"b", 2}};

    SUBCASE("existing keys") {
      auto func = func_from_map(map);
      CHECK(func("a") == 1);
      CHECK(func("b") == 2);
    }

    SUBCASE("missing key") {
      auto func = func_from_map(map);
      CHECK_THROWS(func("c"));
    }

    SUBCASE("empty map") {
      std::unordered_map<std::string, int> map = {};
      auto func = func_from_map(map);
      CHECK_THROWS(func("a"));
    }
  }
}
