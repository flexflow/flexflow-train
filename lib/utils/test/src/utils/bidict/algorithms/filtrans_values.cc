#include "utils/bidict/algorithms/filtrans_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filtrans_values(bidict<K, V>, F)") {
    bidict<int, std::string> dict = {
      {1, "one"},
      {2, "two"},
    };

    bidict<int, int> result = filtrans_values(
        dict, [](std::string const &v) -> std::optional<int> {
          if (v == "two") {
            return std::nullopt;
          } else {
            return v.size() + 1;
          }
        });
    bidict<int, int> correct = {
        {1, 4},
    };
    CHECK(result == correct);
  }
}
