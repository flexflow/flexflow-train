#include <doctest/doctest.h>
#include "op-attrs/ff_ordered/reversed.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reversed(FFOrdered<T>)") {
    SUBCASE("non-empty input") {
      FFOrdered<int> input = {1, 2, 3, 2};

      FFOrdered<int> result = reversed(input);
      FFOrdered<int> correct = {2, 3, 2, 1};

      CHECK(result == correct);
    }

    SUBCASE("empty input") {
      FFOrdered<int> input = {};

      FFOrdered<int> result = reversed(input);
      FFOrdered<int> correct = {};

      CHECK(result == correct);
    }
  }
}
