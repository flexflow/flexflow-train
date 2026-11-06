#include "utils/bidict/algorithms/values.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>
#include <map>
#include <set>
#include <string>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("values(bidict)") {
    bidict<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    std::set<std::string> result = values(m);
    std::set<std::string> correct = {"one", "two", "three"};
    CHECK(result == correct);
  }
}
