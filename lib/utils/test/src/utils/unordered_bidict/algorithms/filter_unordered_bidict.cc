#include "utils/unordered_bidict/algorithms/filter_unordered_bidict.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filter_unordered_bidict") {
    unordered_bidict<int, std::string> b = {
        {1, "one"},
        {2, "two"},
        {3, "three"},
        {4, "four"},
    };

    auto filter_func = [](int k, std::string const &v) -> bool {
      return (k % 2) == 0 || v.size() == 3;
    };

    unordered_bidict<int, std::string> result =
        filter_unordered_bidict(b, filter_func);
    unordered_bidict<int, std::string> correct = {
        {1, "one"},
        {2, "two"},
        {4, "four"},
    };

    CHECK(result == correct);
  }
}
