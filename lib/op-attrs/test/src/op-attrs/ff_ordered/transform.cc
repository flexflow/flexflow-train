#include "op-attrs/ff_ordered/transform.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform(FFOrdered<T>, F)") {
    SUBCASE("input is empty") {
      FFOrdered<std::string> input = {};

      FFOrdered<int> result = transform(input, [](std::string const &) -> int {
        CHECK(false);
        return 0;
      });
      FFOrdered<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("input is not empty") {
      FFOrdered<int> input = FFOrdered{2, 1, 2, 5};

      FFOrdered<std::string> result =
          transform(input, [](int x) { return fmt::to_string(x); });
      FFOrdered<std::string> correct = FFOrdered<std::string>{
          "2",
          "1",
          "2",
          "5",
      };

      CHECK(result == correct);
    }
  }
}
