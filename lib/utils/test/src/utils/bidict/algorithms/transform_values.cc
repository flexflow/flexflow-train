#include "utils/bidict/algorithms/transform_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform_values(bidict<K, V>, F)") {
    bidict<int, std::string> dict = {
      {1, "one"},
      {2, "two"},
    };

    bidict<int, std::string> result =
        transform_values(dict, [](std::string const &v) { return v + "a"; });
    bidict<int, std::string> correct = {
        {1, "onea"},
        {2, "twoa"},
    };
    CHECK(result == correct);
  }
}
