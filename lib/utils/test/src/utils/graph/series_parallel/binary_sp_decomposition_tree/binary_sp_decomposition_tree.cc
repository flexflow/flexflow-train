#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_tree_height") {
    auto make_series_split = [](BinarySPDecompositionTree const &lhs,
                                BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs,
                                  BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](size_t n) {
      return BinarySPDecompositionTree{Node{n}};
    };

    SUBCASE("leaf only") {
      BinarySPDecompositionTree input = make_leaf(5);

      nonnegative_int result = get_tree_height(input);
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("series split") {
      BinarySPDecompositionTree input = make_series_split(
          make_series_split(make_series_split(make_leaf(0), make_leaf(1)),
                            make_series_split(make_leaf(2), make_leaf(3))),
          make_series_split(make_leaf(4), make_leaf(5)));

      nonnegative_int result = get_tree_height(input);
      nonnegative_int correct = 3_n;

      CHECK(result == correct);
    }

    SUBCASE("parallel split") {
      BinarySPDecompositionTree input = make_parallel_split(
          make_leaf(4),
          make_parallel_split(
              make_parallel_split(make_leaf(3), make_leaf(1)),
              make_parallel_split(
                  make_parallel_split(make_leaf(2), make_leaf(3)),
                  make_leaf(3))));

      nonnegative_int result = get_tree_height(input);
      nonnegative_int correct = 4_n;

      CHECK(result == correct);
    }

    SUBCASE("mixed") {
      BinarySPDecompositionTree input = make_parallel_split(
          make_leaf(4),
          make_parallel_split(
              make_series_split(make_leaf(3), make_leaf(1)),
              make_parallel_split(
                  make_series_split(
                      make_leaf(2),
                      make_series_split(make_leaf(4), make_leaf(3))),
                  make_leaf(3))));

      nonnegative_int result = get_tree_height(input);
      nonnegative_int correct = 5_n;

      CHECK(result == correct);
    }
  }
}
