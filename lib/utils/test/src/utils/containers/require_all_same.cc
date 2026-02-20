#include "utils/containers/require_all_same.h"
#include "test/utils/doctest/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("require_all_same") {
    SUBCASE("input is empty") {
      std::vector<int> input = {};

      std::optional<int> result = require_all_same(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input is not empty") {
      SUBCASE("input is all the same") {
        std::vector<int> input = {2, 2, 2};

        std::optional<int> result = require_all_same(input);
        std::optional<int> correct = 2;

        CHECK(result == correct);
      }

      SUBCASE("input is not all the same") {
        std::vector<int> input = {2, 2, 3, 2};

        CHECK_THROWS(require_all_same(input));
      }
    }
  }
}
