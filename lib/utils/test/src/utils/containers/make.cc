#include "utils/containers/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make") {
    auto f = make<int>();

    int result = f(true);
    int correct = 1;

    CHECK(result == correct);
  }
}
