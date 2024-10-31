#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("nonnegative_int initialization") {
    SUBCASE("positive int initialization") {
      CHECK_NOTHROW(nonnegative_int(1));
    }

    SUBCASE("zero initialization") {
      CHECK_NOTHROW(nonnegative_int(0));
    }

    SUBCASE("negative int initialization") {
      CHECK_THROWS(nonnegative_int(-1));
    }
  }

  TEST_CASE("nonnegative_int comparisons") {
    SUBCASE("equality") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_1 == nn_int_2) == false);
      CHECK((nn_int_1 == nn_int_1) == true);
      CHECK((nn_int_1 == 2) == false);
      CHECK((nn_int_1 == 1) == true);
      CHECK((1 == nn_int_2) == false);
      CHECK((1 == nn_int_1) == true);
    }

    SUBCASE("not equals") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_1 != nn_int_2) == true);
      CHECK((nn_int_1 != nn_int_1) == false);
      CHECK((nn_int_1 != 2) == true);
      CHECK((nn_int_1 != 1) == false);
      CHECK((1 != nn_int_2) == true);
      CHECK((1 != nn_int_1) == false);
    }

    SUBCASE("less than") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_1 < nn_int_2) == true);
      CHECK((nn_int_2 < nn_int_1) == false);
      CHECK((nn_int_1 < 2) == true);
      CHECK((nn_int_2 < 1) == false);
      CHECK((1 < nn_int_2) == true);
      CHECK((2 < nn_int_1) == false);
    }

    SUBCASE("less than or equal to") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_1 <= nn_int_2) == true);
      CHECK((nn_int_2 <= nn_int_1) == false);
      CHECK((nn_int_1 <= 2) == true);
      CHECK((nn_int_2 <= 1) == false);
      CHECK((1 <= nn_int_2) == true);
      CHECK((2 <= nn_int_1) == false);
    }

    SUBCASE("greater than") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_1 > nn_int_2) == false);
      CHECK((nn_int_2 > nn_int_1) == true);
      CHECK((nn_int_1 > 2) == false);
      CHECK((nn_int_2 > 1) == true);
      CHECK((1 > nn_int_2) == false);
      CHECK((2 > nn_int_1) == true);
    }

    SUBCASE("greater than or equal to") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_1 >= nn_int_2) == false);
      CHECK((nn_int_2 >= nn_int_1) == true);
      CHECK((nn_int_1 >= 2) == false);
      CHECK((nn_int_2 >= 1) == true);
      CHECK((1 >= nn_int_2) == false);
      CHECK((2 >= nn_int_1) == true);
    }
  }

  TEST_CASE("nonnegative_int arithmetic operations") {
    SUBCASE("addition") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_2 == nn_int_1 + 1) == true);
      CHECK((nn_int_2 == 1 + nn_int_1) == true);
      CHECK((nn_int_2 == nn_int_1 + nn_int_1) == true);
      nn_int_1 += nn_int_1;
      CHECK((2 == nn_int_1) == true);
      nn_int_1 += 1;
      CHECK((3 == nn_int_1) == true);
      CHECK((++nn_int_1) == 4);
      nn_int_1++;
      CHECK((5 == nn_int_1) == true);
    }

    SUBCASE("subtraction") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK((nn_int_1 == nn_int_2 - 1) == true);
      CHECK((nn_int_1 == 2 - nn_int_1) == true);
      CHECK((nn_int_1 == nn_int_2 - nn_int_1) == true);
      nonnegative_int nn_int_5 = nonnegative_int(5);
      nn_int_5 -= nn_int_1;
      CHECK((4 == nn_int_5) == true);
      nn_int_5 -= 1;
      CHECK((3 == nn_int_5) == true);
      CHECK((--nn_int_5) == 2);
      nn_int_5--;
      CHECK((1 == nn_int_5) == true);
    }

    SUBCASE("subtraction negative result") {
      nonnegative_int nn_int_1 = nonnegative_int(1);
      nonnegative_int nn_int_2 = nonnegative_int(2);
      CHECK_THROWS(nn_int_1 - 2);
      CHECK_THROWS(1 - nn_int_2);
      CHECK_THROWS(nn_int_1 - nn_int_2);
      CHECK_THROWS(nn_int_1 -= 2);
      CHECK_THROWS(nn_int_1 -= nn_int_2);
      nonnegative_int nn_int_0 = nonnegative_int(0);
      CHECK_THROWS(nn_int_0--);
      CHECK_THROWS(--nn_int_0);
    }

    SUBCASE("multiplication") {
      nonnegative_int nn_int_2 = nonnegative_int(2);
      nonnegative_int nn_int_3 = nonnegative_int(3);
      nonnegative_int nn_int_6 = nonnegative_int(6);
      CHECK((nn_int_6 == (nn_int_2 * 3)) == true);
      CHECK((nn_int_6 == (2 * nn_int_3)) == true);
      CHECK((nn_int_6 == (nn_int_2 * nn_int_3)) == true);
      nn_int_2 *= 3;
      CHECK((nn_int_6 == nn_int_2) == true);
      nn_int_2 *= nn_int_3;
      CHECK((18 == nn_int_2) == true);
    }

    SUBCASE("division") {
      nonnegative_int nn_int_2 = nonnegative_int(2);
      nonnegative_int nn_int_3 = nonnegative_int(3);
      nonnegative_int nn_int_6 = nonnegative_int(6);
      CHECK((nn_int_3 == (nn_int_6 / 2)) == true);
      CHECK((nn_int_3 == (6 / nn_int_2)) == true);
      CHECK((nn_int_3 == (nn_int_6 / nn_int_2)) == true);
      nn_int_6 /= 3;
      CHECK((nn_int_6 == nn_int_2) == true);
      nn_int_6 /= nn_int_2;
      CHECK((1 == nn_int_6) == true);
    }

    SUBCASE("divide by 0") {
      nonnegative_int nn_int_0 = nonnegative_int(0);
      nonnegative_int nn_int_1 = nonnegative_int(1);
      CHECK_THROWS(nn_int_1 / 0);
      CHECK_THROWS(1 / nn_int_0);
      CHECK_THROWS(nn_int_1 / nn_int_0);
      CHECK_THROWS(nn_int_1 /= 0);
      CHECK_THROWS(nn_int_1 /= nn_int_0);
    }
  }

  TEST_CASE("nonnegative_int adl_serializer") {
    SUBCASE("to_json") {
      nonnegative_int input = nonnegative_int{5};

      nlohmann::json result = input;
      nlohmann::json correct = 5;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      nlohmann::json input = 5;

      nonnegative_int result = input.template get<nonnegative_int>();
      nonnegative_int correct = nonnegative_int{5};

      CHECK(result == correct);
    }
  }
}
