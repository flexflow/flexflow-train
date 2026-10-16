#include "utils/containers/set_filtrans.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("set_filtrans(std::vector<In>, F)") {
    SUBCASE("input is not empty") {
      std::vector<int> input = {1, 2, 3, 2, 4};
      std::set<std::string> result =
          set_filtrans(input, [](int x) -> std::optional<std::string> {
            if ((x % 2) == 0) {
              return std::to_string(x);
            } else {
              return std::nullopt;
            }
          });

      std::set<std::string> correct = {"2", "2", "4"};

      CHECK(result == correct);
    }

    SUBCASE("input is empty") {
      std::vector<int> input = {};
      std::set<std::string> result = set_filtrans(
          input, [](int x) -> std::optional<std::string> { PANIC(); });

      std::set<std::string> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("set_filtrans(std::set<In>, F)") {
    SUBCASE("input is not empty") {
      std::set<int> input = {1, 2, 3, 4};
      std::set<std::string> result =
          set_filtrans(input, [](int x) -> std::optional<std::string> {
            if ((x % 2) == 0) {
              return std::to_string(x);
            } else {
              return std::nullopt;
            }
          });

      std::set<std::string> correct = {"2", "4"};

      CHECK(result == correct);
    }

    SUBCASE("input is empty") {
      std::set<int> input = {};
      std::set<std::string> result = set_filtrans(
          input, [](int x) -> std::optional<std::string> { PANIC(); });

      std::set<std::string> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("set_filtrans(std::unordered_set<In>, F)") {
    SUBCASE("input is not empty") {
      std::unordered_set<int> input = {1, 2, 3, 4};
      std::set<std::string> result =
          set_filtrans(input, [](int x) -> std::optional<std::string> {
            if ((x % 2) == 0) {
              return std::to_string(x);
            } else {
              return std::nullopt;
            }
          });

      std::set<std::string> correct = {"2", "4"};

      CHECK(result == correct);
    }

    SUBCASE("input is empty") {
      std::unordered_set<int> input = {};
      std::set<std::string> result = set_filtrans(
          input, [](int x) -> std::optional<std::string> { PANIC(); });

      std::set<std::string> correct = {};

      CHECK(result == correct);
    }
  }
}
