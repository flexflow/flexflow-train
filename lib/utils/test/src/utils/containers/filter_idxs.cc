#include "utils/containers/filter_idxs.h"
#include <doctest/doctest.h>
#include <string>
#include "test/utils/doctest/fmt/vector.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filter_idxs") {
    std::vector<std::string> input = {"hello", "world", "!"};

    std::vector<std::string> result = filter_idxs(input, [](nonnegative_int idx) { return idx % 2_n == 0; });
    std::vector<std::string> correct = {"hello", "!"};

    CHECK(result == correct);
  }
}
