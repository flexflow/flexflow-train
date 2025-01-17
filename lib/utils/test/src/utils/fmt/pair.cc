#include "utils/fmt/pair.h"
#include <doctest/doctest.h>
#include <sstream>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::pair<int, int>)") {
    std::pair<int, int> input = {3, 5};
    std::string result = fmt::to_string(input);
    std::string correct = "{3, 5}";
    CHECK(result == correct);
  }

  TEST_CASE("operator<<(ostream &, std::pair<int, int>)") {
    std::pair<int, int> input = {3, 5};

    std::ostringstream oss;
    oss << input;
    std::string result = oss.str();

    std::string correct = "{3, 5}";
    CHECK(result == correct);
  }
}
