#include "op-attrs/ff_ordered/ff_ordered.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("Arbitrary<FFOrdered<T>> with T=", T, int, double, char) {
    RC_SUBCASE([](FFOrdered<T>) {});
  }
}
