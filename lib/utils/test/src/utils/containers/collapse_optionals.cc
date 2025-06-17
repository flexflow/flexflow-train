#include <doctest/doctest.h>
#include "utils/containers/collapse_optionals.h"
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("collapse_optionals(std::optional<std::optional<T>>)") {
    SUBCASE("returns the value if the input has value") {
      std::optional<std::optional<int>> input = 8;

      std::optional<int> result = collapse_optionals(input);
      std::optional<int> correct = 8;

      CHECK(result == correct);
    }

    SUBCASE("returns nullopt if the input is wrapped nullopt") {
      std::optional<std::optional<int>> input = std::optional<int>{std::nullopt};

      std::optional<int> result = collapse_optionals(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("returns nullopt if the input is unwrapped nullopt") {
      std::optional<std::optional<int>> input = std::optional<std::optional<int>>{std::nullopt};

      std::optional<int> result = collapse_optionals(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
