#include "utils/int_ge_two/int_ge_two.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("int_ge_two") {
    SUBCASE("constructor") {
      SUBCASE("throws if value is less than 2") {
        CHECK_THROWS(int_ge_two{1});
        CHECK_THROWS(int_ge_two{0});
        CHECK_THROWS(int_ge_two{-1});
      }

      SUBCASE("wraps the value if >= 2") {
        int_ge_two x = int_ge_two{2};

        CHECK(x.int_from_int_ge_two() == 2);
      }
    }

    SUBCASE("positive_int *= int_ge_two") {
      positive_int x = 3_p;
      x *= int_ge_two{4};

      positive_int correct = 12_p;

      CHECK(x == correct);
    }

    SUBCASE("nonnegative_int *= int_ge_two") {
      SUBCASE("starting value is zero") {
        nonnegative_int x = 0_n;
        x *= int_ge_two{4};

        nonnegative_int correct = 0_n;

        CHECK(x == correct);
      }

      SUBCASE("starting value is nonzero") {
        nonnegative_int x = 5_n;
        x *= int_ge_two{4};

        nonnegative_int correct = 20_n;

        CHECK(x == correct);
      }
    }

    SUBCASE("int_ge_two / int_ge_two") {
      nonnegative_int result = int_ge_two{2} / int_ge_two{8};
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("positive_int / int_ge_two") {
      nonnegative_int result = positive_int{4} / int_ge_two{2};
      nonnegative_int correct = 2_n;

      CHECK(result == correct);
    }

    SUBCASE("nonnegative_int / int_ge_two") {
      nonnegative_int result = nonnegative_int{0} / int_ge_two{2};
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("positive_int % int_ge_two") {
      nonnegative_int result = positive_int{4} % int_ge_two{2};
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("nonnegative_int % int_ge_two") {
      nonnegative_int result = nonnegative_int{3} % int_ge_two{2};
      nonnegative_int correct = 1_n;

      CHECK(result == correct);
    }
  }
}
