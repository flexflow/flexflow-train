#include "utils/containers/lift_optional_through_map.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("lift_optional_through_map") {
    SUBCASE("throws if only some of the values are nullopt") {
      std::unordered_map<int, std::optional<std::string>> input = {
          {1, std::nullopt},
          {2, "two"},
      };

      CHECK_THROWS(lift_optional_through_map(input));
    }

    SUBCASE("returns nullopt if all of the values are nullopt") {
      std::unordered_map<int, std::optional<std::string>> input = {
          {1, std::nullopt},
          {2, std::nullopt},
      };

      std::optional<std::unordered_map<int, std::string>> result =
          lift_optional_through_map(input);

      std::optional<std::unordered_map<int, std::string>> correct =
          std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("returns the map if all of the values are not nullopt") {
      std::unordered_map<int, std::optional<std::string>> input = {
          {1, "one"},
          {2, "two"},
      };

      std::optional<std::unordered_map<int, std::string>> result =
          lift_optional_through_map(input);

      std::optional<std::unordered_map<int, std::string>> correct =
          std::unordered_map<int, std::string>{
              {1, "one"},
              {2, "two"},
          };

      CHECK(result == correct);
    }

    SUBCASE("throws if the input is an empty map") {
      std::unordered_map<int, std::optional<std::string>> input = {};

      CHECK_THROWS(lift_optional_through_map(input));
    }
  }
}
