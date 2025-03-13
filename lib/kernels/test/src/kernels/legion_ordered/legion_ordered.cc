#include "kernels/legion_ordered/legion_ordered.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE(
      "Arbitrary<LegionOrdered<T>> with T=", T, int, double, char) {
    RC_SUBCASE([](LegionOrdered<T>) {});
  }
}
