#include "utils/containers/try_get_one_of.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/containers/contains.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("try_get_one_of(std::unordered_set<T>)") {
    SUBCASE("input is empty") {
      std::unordered_set<int> input = {};

      std::optional<int> result = try_get_one_of(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input is non-empty") {
      std::unordered_set<int> input = {1, 2, 3};

      std::optional<int> result = try_get_one_of(input);

      CHECK(result.has_value());
      CHECK(contains(input, result.value()));
    }
  }

  TEST_CASE("try_get_one_of(std::set<T>)") {
    SUBCASE("input is empty") {
      std::set<int> input = {};

      std::optional<int> result = try_get_one_of(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input is non-empty") {
      std::set<int> input = {1, 2, 3};

      std::optional<int> result = try_get_one_of(input);

      CHECK(result.has_value());
      CHECK(contains(input, result.value()));
    }
  }
}
