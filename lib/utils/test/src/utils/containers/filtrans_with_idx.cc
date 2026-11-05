#include "utils/containers/filtrans_with_idx.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filtrans_with_idx") {
    SUBCASE("input is empty") {
      std::vector<int> input = {};

      std::vector<std::string> result = filtrans_with_idx(
          input,
          [&](nonnegative_int idx, int elem) -> std::optional<std::string> {
            PANIC();
          });

      std::vector<std::string> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("function uses index") {
      std::vector<int> input = {2, 1, 2, 3, 5, 4};

      std::vector<nonnegative_int> result = filtrans_with_idx(
          input,
          [&](nonnegative_int idx, int elem) -> std::optional<nonnegative_int> {
            if (idx == 4_n) {
              return std::nullopt;
            } else {
              return idx;
            }
          });

      std::vector<nonnegative_int> correct = {0_n, 1_n, 2_n, 3_n, 5_n};

      CHECK(result == correct);
    }

    SUBCASE("function uses value") {
      std::vector<int> input = {2, 1, 2, 4};

      std::vector<std::string> result = filtrans_with_idx(
          input,
          [&](nonnegative_int idx, int elem) -> std::optional<std::string> {
            return std::to_string(elem);
          });

      std::vector<std::string> correct = {"2", "1", "2", "4"};

      CHECK(result == correct);
    }

    SUBCASE("function combines index and value") {
      std::vector<int> input = {2, 1, 2, 4};

      std::vector<std::pair<nonnegative_int, int>> result = filtrans_with_idx(
          input,
          [&](nonnegative_int idx,
              int elem) -> std::optional<std::pair<nonnegative_int, int>> {
            return std::make_pair(idx, elem);
          });

      std::vector<std::pair<nonnegative_int, int>> correct = {
          {0_n, 2},
          {1_n, 1},
          {2_n, 2},
          {3_n, 4},
      };

      CHECK(result == correct);
    }

    SUBCASE("function always returns nullopt") {
      std::vector<int> input = {2, 1, 2, 4};

      std::vector<std::string> result = filtrans_with_idx(
          input,
          [&](nonnegative_int idx, int elem) -> std::optional<std::string> {
            return std::nullopt;
          });

      std::vector<std::string> correct = {};

      CHECK(result == correct);
    }
  }
}
