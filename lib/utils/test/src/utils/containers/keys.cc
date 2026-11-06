#include "utils/containers/keys.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <string>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("keys(std::map)") {
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "three"}};
    std::set<int> result = keys(m);
    std::set<int> expected = {1, 2, 3};
    CHECK(result == expected);
  }
  TEST_CASE("keys(std::unordered_map)") {
    std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "three"}};
    std::unordered_set<int> result = keys(m);
    std::unordered_set<int> expected = {1, 2, 3};
    CHECK(result == expected);
  }
}
