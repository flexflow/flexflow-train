#include "utils/bidict/algorithms/transform.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform(bidict<K, V>, F)") {
    bidict<int, std::string> dict = {
      {1, "one"},
      {2, "two"},
    };

    bidict<std::string, int> result =
        transform(dict, [](int k, std::string const &v) {
          return std::make_pair(v, k);
        });
    bidict<std::string, int> correct = {
        {"one", 1},
        {"two", 2},
    };

    CHECK(result == correct);
  }
}
