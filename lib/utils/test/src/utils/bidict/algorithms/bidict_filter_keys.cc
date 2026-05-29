#include "utils/bidict/algorithms/bidict_filter_keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict_filter_keys(bidict<K, V>, F)") {
    bidict<int, std::string> dict = {
        {1, "one"},
        {2, "two"},
    };

    bidict<int, std::string> result =
        bidict_filter_keys(dict, [](int k) { return k == 1; });
    bidict<int, std::string> correct = {
        {1, "one"},
    };

    CHECK(result == correct);
  }
}
