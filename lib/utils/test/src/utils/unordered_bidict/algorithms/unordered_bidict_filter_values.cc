#include "utils/unordered_bidict/algorithms/unordered_bidict_filter_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict_filter_values(unordered_bidict<K, V>, F") {
    unordered_bidict<int, std::string> dict = {
        {1, "one"},
        {2, "two"},
    };

    unordered_bidict<int, std::string> result = unordered_bidict_filter_values(
        dict, [](std::string const &v) { return v == "two"; });
    unordered_bidict<int, std::string> correct = {
        {2, "two"},
    };

    CHECK(result == correct);
  }
}
