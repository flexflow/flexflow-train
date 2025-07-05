#include "kernels/legion_ordered/transform.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("transform(LegionOrdered<T>, F)") {
    SUBCASE("input is empty") {
      LegionOrdered<std::string> input = {};

      LegionOrdered<int> result =
          transform(input, [](std::string const &) -> int {
            CHECK(false);
            return 0;
          });
      LegionOrdered<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("input is not empty") {
      LegionOrdered<int> input = LegionOrdered{2, 1, 2, 5};

      LegionOrdered<std::string> result =
          transform(input, [](int x) { return fmt::to_string(x); });
      LegionOrdered<std::string> correct = LegionOrdered<std::string>{
          "2",
          "1",
          "2",
          "5",
      };

      CHECK(result == correct);
    }
  }
}
