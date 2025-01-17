#include "utils/containers/uncurry.h"
#include <doctest/doctest.h>
#include <string>
#include "test/utils/doctest/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("uncurry") {
    auto f = [](int x, std::string const &y) {
      return std::make_pair(y, x);
    };

    std::function<std::pair<std::string, int>(std::pair<int, std::string> const &)>
      result_f = uncurry<int, std::string>(f);

    SUBCASE("has same behavior as f") {
      int x = 1;
      std::string y = "aa";
      std::pair<int, std::string> p = {1, "aa"};

      std::pair<std::string, int> result = result_f(p);
      std::pair<std::string, int> correct = f(x, y);
      CHECK(result == correct);
    }
  }
}
