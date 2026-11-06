#include "utils/unordered_bidict/algorithms/unordered_bidict_transform_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict_transform_values(unordered_bidict<K, V>, F)") {
    unordered_bidict<int, std::string> dict = {
        {1, "one"},
        {2, "two"},
    };

    unordered_bidict<int, std::string> result = unordered_bidict_transform_values(
        dict, [](std::string const &v) { return v + "a"; });
    unordered_bidict<int, std::string> correct = {
        {1, "onea"},
        {2, "twoa"},
    };
    CHECK(result == correct);
  }
}
