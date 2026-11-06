#include "utils/unordered_bidict/algorithms/values.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("values(unordered_bidict)") {
    unordered_bidict<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    std::unordered_set<std::string> result = values(m);
    std::unordered_set<std::string> correct = {"one", "two", "three"};
    CHECK(result == correct);
  }
}
