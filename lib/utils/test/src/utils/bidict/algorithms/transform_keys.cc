#include "utils/bidict/algorithms/transform_keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform_keys(bidict<K, V>, F)") {
    bidict<int, std::string> dict = {
      {1, "one"},
      {2, "two"},
    };

    bidict<std::string, std::string> result = transform_keys(dict, [](int k) {
      std::ostringstream oss;
      oss << k;
      return oss.str();
    });
    bidict<std::string, std::string> correct = {
        {"1", "one"},
        {"2", "two"},
    };

    CHECK(result == correct);
  }
}
