#include <doctest/doctest.h>
#include "utils/containers/make_counter_func.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_counter_func") {
    SUBCASE("with starting value") {
      std::function<int()> ctr = make_counter_func(2);
      CHECK(ctr() == 2);
      CHECK(ctr() == 3);
    }

    SUBCASE("without starting value") {
      std::function<int()> ctr = make_counter_func();
      CHECK(ctr() == 0);
      CHECK(ctr() == 1);
    }

    SUBCASE("copying") {
      std::function<int()> ctr = make_counter_func(4);
      CHECK(ctr() == 4);

      std::function<int()> ctr_2 = ctr;
      CHECK(ctr_2() == 5);
      CHECK(ctr() == 6);
      CHECK(ctr_2() == 7);
    }
  }
}
