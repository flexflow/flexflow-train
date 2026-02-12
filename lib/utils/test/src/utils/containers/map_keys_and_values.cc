#include "utils/containers/map_keys_and_values.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_keys_and_values") {
    SUBCASE("Distinct keys after transformation") {
      std::unordered_map<int, std::string> m = {{1, "one"}, {2, "three"}};
      auto fk = [](int x) { return x * x; };
      auto fv = [](std::string const &s) { return s.size(); };
      std::unordered_map<int, size_t> result = map_keys_and_values(m, fk, fv);
      std::unordered_map<int, size_t> correct = {{1, 3}, {4, 5}};
      CHECK(correct == result);
    }

    SUBCASE("Non-distinct keys after transformation") {
      std::unordered_map<int, std::string> m = {
          {1, "one"}, {2, "two"}, {-1, "minus one"}};
      auto fk = [](int x) { return std::abs(x); };
      auto fv = [](std::string const &s) { return s.size(); };
      CHECK_THROWS(map_keys_and_values(m, fk, fv));
    }
  }
}
