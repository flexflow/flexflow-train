#include "utils/containers/values.h"
#include "test/utils/doctest/fmt/multiset.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include <doctest/doctest.h>
#include <map>
#include <string>
#include <unordered_map>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("values(std::map)") {
    std::map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}, {33, "three"}};
    std::multiset<std::string> result = values(m);
    std::multiset<std::string> correct = {"one", "two", "three", "three"};
    CHECK(result == correct);
  }
  TEST_CASE("values(std::unordered_map)") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}, {33, "three"}};
    std::unordered_multiset<std::string> result = values(m);
    std::unordered_multiset<std::string> correct = {
        "one", "two", "three", "three"};
    CHECK(result == correct);
  }
}
