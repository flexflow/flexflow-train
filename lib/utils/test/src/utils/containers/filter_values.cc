#include "utils/containers/filter_values.h"
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>
#include <map>
#include <string>
#include <unordered_map>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filter_values(std::map)") {
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "blah"}};
    auto f = [](std::string const &x) { return x.size() % 2 == 1; };
    std::map<int, std::string> result = filter_values(m, f);
    std::map<int, std::string> correct = {{1, "one"}, {2, "two"}};
    CHECK(result == correct);
  }
  TEST_CASE("filter_values(std::unordered_map)") {
    std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "blah"}};
    auto f = [](std::string const &x) { return x.size() % 2 == 1; };
    std::unordered_map<int, std::string> result = filter_values(m, f);
    std::unordered_map<int, std::string> correct = {{1, "one"}, {2, "two"}};
    CHECK(result == correct);
  }
}
