#include "utils/bidict/algorithms/filter_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filter_values(bidict<K, V>, F") {
    bidict<int, std::string> dict = {
      {1, "one"},
      {2, "two"},
    };

    bidict<int, std::string> result =
        filter_values(dict, [](std::string const &v) { return v == "two"; });
    bidict<int, std::string> correct = {
        {2, "two"},
    };

    CHECK(result == correct);
  }
}
