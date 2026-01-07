#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unstructured_relation_from_bidict") {
    bidict<int, std::string> input = {
        {1, "one"},
        {2, "two"},
    };

    std::unordered_set<std::pair<int, std::string>> result =
        unstructured_relation_from_bidict(input);
    std::unordered_set<std::pair<int, std::string>> correct = {
        {1, "one"},
        {2, "two"},
    };

    CHECK(result == correct);
  }
}
