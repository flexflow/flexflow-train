#include "utils/containers/lookup_in_map.h"
#include <doctest/doctest.h>
#include <functional>
#include <string>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("lookup_in_map") {

    std::unordered_map<std::string, int> map = {{"a", 1}, {"b", 2}};

    SUBCASE("existing keys") {
      std::function<int(std::string const &)> func = lookup_in_map(map);
      CHECK(func("a") == 1);
      CHECK(func("b") == 2);
    }

    SUBCASE("missing key") {
      std::function<int(std::string const &)> func = lookup_in_map(map);
      CHECK_THROWS(func("c"));
    }

    SUBCASE("empty map") {
      std::unordered_map<std::string, int> map = {};
      std::function<int(std::string const &)> func = lookup_in_map(map);
      CHECK_THROWS(func("a"));
    }
  }
}
