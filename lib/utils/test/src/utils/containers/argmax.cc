#include "utils/containers/argmax.h"
#include "utils/containers/contains.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("argmax") {
    SUBCASE("non-empty container") {
      std::vector<int> input = {-5, -1, 2, 4};
      int result = argmax(input, [](int x) { return -x * x; });
      CHECK(result == -1);
    }

    SUBCASE("empty container") {
      std::vector<int> input = {};
      CHECK_THROWS(argmax(input, [](int x) { return x; }));
    }

    SUBCASE("ties") {
      std::unordered_set<int> input = {-1, 1, 2};
      int result = argmax(input, [](int x) { return -(x * x); });
      CHECK(contains(std::unordered_set<int>{-1, 1}, result));
    }
  }
}
