#include "utils/unordered_bidict/algorithms/unordered_bidict_transform_keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict_transform_keys(unordered_bidict<K, V>, F)") {
    unordered_bidict<int, std::string> dict = {
        {1, "one"},
        {2, "two"},
    };

    unordered_bidict<std::string, std::string> result =
        unordered_bidict_transform_keys(dict, [](int k) {
          std::ostringstream oss;
          oss << k;
          return oss.str();
        });
    unordered_bidict<std::string, std::string> correct = {
        {"1", "one"},
        {"2", "two"},
    };

    CHECK(result == correct);
  }
}
