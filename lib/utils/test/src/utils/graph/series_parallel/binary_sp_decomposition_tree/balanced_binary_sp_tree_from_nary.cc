#include "utils/graph/series_parallel/binary_sp_decomposition_tree/balanced_binary_sp_tree_from_nary.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/contains.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/nary_sp_tree_from_binary.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("balanced_binary_sp_tree_from_nary") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};
    Node n5 = Node{5};

    auto make_series_split = [](BinarySPDecompositionTree const &lhs,
                                BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs,
                                  BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) { return BinarySPDecompositionTree{n}; };

    SUBCASE("only node") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{n1};

      BinarySPDecompositionTree result =
          balanced_binary_sp_tree_from_nary(input);
      BinarySPDecompositionTree correct = make_leaf(n1);

      CHECK(result == correct);
    }

    SUBCASE("only serial") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{n1, n2, n3, n4}},
      };

      BinarySPDecompositionTree result =
          balanced_binary_sp_tree_from_nary(input);

      BinarySPDecompositionTree correct =
          make_series_split(make_series_split(make_leaf(n1), make_leaf(n2)),
                            make_series_split(make_leaf(n3), make_leaf(n4)));

      CHECK(result == correct);
    }

    SUBCASE("only parallel") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          ParallelSplit{{n1, n2, n3, n4}},
      };
      BinarySPDecompositionTree result =
          balanced_binary_sp_tree_from_nary(input);

      int result_height = get_tree_height(result);
      int expected_height = 2;
      CHECK(result_height == expected_height);

      std::unordered_multiset<Node> result_nodes = get_nodes(result);
      std::unordered_multiset<Node> expected_nodes = {n1, n2, n3, n4};
      CHECK(result_nodes == expected_nodes);
    }

    SUBCASE("nested") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          ParallelSplit{{SeriesSplit{{n1, n2, n3, n4}}, n5}},
      };

      BinarySPDecompositionTree result =
          balanced_binary_sp_tree_from_nary(input);

      BinarySPDecompositionTree balanced_series =
          make_series_split(make_series_split(make_leaf(n1), make_leaf(n2)),
                            make_series_split(make_leaf(n3), make_leaf(n4)));

      std::unordered_set<BinarySPDecompositionTree> corrects = {
          make_parallel_split(balanced_series, make_leaf(n5)),
          make_parallel_split(make_leaf(n5), balanced_series)};

      CHECK(contains(corrects, result));
    }
  }
}
