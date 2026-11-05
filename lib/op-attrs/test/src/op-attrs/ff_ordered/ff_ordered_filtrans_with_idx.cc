#include "op-attrs/ff_ordered/ff_ordered_filtrans_with_idx.h"
#include "test/utils/doctest/fmt/pair.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ff_ordered_filtrans_with_idx") {
    SUBCASE("input is empty") {
      FFOrdered<int> input = {};

      FFOrdered<std::string> result = ff_ordered_filtrans_with_idx(
          input, [&](ff_dim_t idx, int elem) -> std::optional<std::string> {
            PANIC();
          });

      FFOrdered<std::string> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("function uses index") {
      FFOrdered<int> input = FFOrdered<int>{2, 1, 2, 3, 5, 4};

      FFOrdered<ff_dim_t> result = ff_ordered_filtrans_with_idx(
          input, [&](ff_dim_t idx, int elem) -> std::optional<ff_dim_t> {
            if (idx == ff_dim_t{4_n}) {
              return std::nullopt;
            } else {
              return idx;
            }
          });

      FFOrdered<ff_dim_t> correct = FFOrdered<ff_dim_t>{
          ff_dim_t{0_n},
          ff_dim_t{1_n},
          ff_dim_t{2_n},
          ff_dim_t{3_n},
          ff_dim_t{5_n},
      };

      CHECK(result == correct);
    }

    SUBCASE("function uses value") {
      FFOrdered<int> input = FFOrdered<int>{2, 1, 2, 4};

      FFOrdered<std::string> result = ff_ordered_filtrans_with_idx(
          input, [&](ff_dim_t idx, int elem) -> std::optional<std::string> {
            return std::to_string(elem);
          });

      FFOrdered<std::string> correct =
          FFOrdered<std::string>{"2", "1", "2", "4"};

      CHECK(result == correct);
    }

    SUBCASE("function combines index and value") {
      FFOrdered<int> input = FFOrdered<int>{2, 1, 2, 4};

      FFOrdered<std::pair<ff_dim_t, int>> result = ff_ordered_filtrans_with_idx(
          input,
          [&](ff_dim_t idx,
              int elem) -> std::optional<std::pair<ff_dim_t, int>> {
            return std::make_pair(idx, elem);
          });

      FFOrdered<std::pair<ff_dim_t, int>> correct =
          FFOrdered<std::pair<ff_dim_t, int>>{
              {ff_dim_t{0_n}, 2},
              {ff_dim_t{1_n}, 1},
              {ff_dim_t{2_n}, 2},
              {ff_dim_t{3_n}, 4},
          };

      CHECK(result == correct);
    }

    SUBCASE("function always returns nullopt") {
      FFOrdered<int> input = FFOrdered<int>{2, 1, 2, 4};

      FFOrdered<std::string> result = ff_ordered_filtrans_with_idx(
          input, [&](ff_dim_t idx, int elem) -> std::optional<std::string> {
            return std::nullopt;
          });

      FFOrdered<std::string> correct = FFOrdered<std::string>{};

      CHECK(result == correct);
    }
  }
}
