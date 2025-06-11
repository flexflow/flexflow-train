#include "utils/positive_int/ceildiv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ceildiv(positive_int, positive_int)") {
    SUBCASE("divides evenly") {
      positive_int numerator = 12_p;
      positive_int denominator = 3_p;

      positive_int result = ceildiv(numerator, denominator);
      positive_int correct = 4_p;

      CHECK(result == correct);
    }

    SUBCASE("does not divide evenly") {
      positive_int numerator = 17_p;
      positive_int denominator = 4_p;

      positive_int result = ceildiv(numerator, denominator);
      positive_int correct = 5_p;

      CHECK(result == correct);
    }
  }
}
