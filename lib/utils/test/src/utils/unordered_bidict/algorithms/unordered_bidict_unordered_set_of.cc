#include "utils/unordered_bidict/algorithms/unordered_bidict_unordered_set_of.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict_unordered_set_of(unordered_bidict<K, V>)") {
    unordered_bidict<int, std::string> b = {
        {1, "one"},
        {2, "two"},
    };

    std::unordered_set<std::pair<int, std::string>> result =
        unordered_bidict_unordered_set_of(b);

    std::unordered_set<std::pair<int, std::string>> correct = {
        {1, "one"},
        {2, "two"},
    };

    CHECK(result == correct);
  }
}
