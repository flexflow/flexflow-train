#include "utils/containers/multiset_of.h"
#include "test/utils/doctest/fmt/multiset.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("multiset_of") {
    std::vector<int> input = {1, 2, 3, 3, 2, 3};
    std::multiset<int> result = multiset_of(input);
    std::multiset<int> correct = {1, 2, 3, 3, 2, 3};
    CHECK(result == correct);
  }
}
