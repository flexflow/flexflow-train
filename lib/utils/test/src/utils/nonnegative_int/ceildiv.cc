#include "utils/nonnegative_int/ceildiv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ceildiv(nonnegative_int, nonnegative_int)") {
    SUBCASE("divides evenly") {
      nonnegative_int numerator = 12_n;
      nonnegative_int denominator = 3_n;

      nonnegative_int result = ceildiv(numerator, denominator);
      nonnegative_int correct = 4_n;

      CHECK(result == correct);
    }

    SUBCASE("does not divide evenly") {
      nonnegative_int numerator = 17_n;
      nonnegative_int denominator = 4_n;

      nonnegative_int result = ceildiv(numerator, denominator);
      nonnegative_int correct = 5_n;

      CHECK(result == correct);
    }

    SUBCASE("denominator is zero") {
      nonnegative_int numerator = 15_n;
      nonnegative_int denominator = 0_n;

      CHECK_THROWS(ceildiv(numerator, denominator));
    }

    SUBCASE("numerator is zero") {
      nonnegative_int numerator = 0_n;
      nonnegative_int denominator = 1_n;

      nonnegative_int result = ceildiv(numerator, denominator);
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("denominator and numerator are zero") {
      nonnegative_int numerator = 0_n;
      nonnegative_int denominator = 0_n;

      CHECK_THROWS(ceildiv(numerator, denominator));
    }
  }
}
