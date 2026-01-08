#include "utils/containers/foldl1.h"
#include "utils/containers/get_only.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/series_parallel/parallel_split.dtg.h"
#include "utils/graph/series_parallel/series_split.dtg.h"
#include "utils/overload.h"
#include <functional>
#include <optional>

namespace FlexFlow {

BinarySPDecompositionTree
    balanced_binary_sp_tree_from_nary(SeriesParallelDecomposition const &nary) {
  std::function<BinarySPDecompositionTree(
      std::variant<ParallelSplit, Node> const &)>
      from_series_child;
  std::function<BinarySPDecompositionTree(
      std::variant<SeriesSplit, Node> const &)>
      from_parallel_child;

  std::function<BinarySPDecompositionTree(ParallelSplit const &p)>
      from_parallel;
  std::function<BinarySPDecompositionTree(SeriesSplit const &p)> from_series;

  auto from_node = [](Node const &n) -> BinarySPDecompositionTree {
    return BinarySPDecompositionTree{n};
  };

  from_parallel = [&](ParallelSplit const &s) -> BinarySPDecompositionTree {
    std::vector<std::variant<SeriesSplit, Node>> children =
        vector_of(s.get_children());
    if (children.size() == 1) {
      return from_parallel_child(get_only(children));
    } else if (children.size() == 2) {
      return BinarySPDecompositionTree{BinaryParallelSplit{
          from_parallel_child(children[0]), from_parallel_child(children[1])}};
    }

    auto s1 = unordered_multiset_of(slice(children, 0, children.size() / 2));
    auto s2 = unordered_multiset_of(
        slice(children, children.size() / 2, std::nullopt));

    return BinarySPDecompositionTree{BinaryParallelSplit{
        from_parallel(ParallelSplit{s1}), from_parallel(ParallelSplit{s2})}};
  };

  from_series = [&](SeriesSplit const &s) -> BinarySPDecompositionTree {
    std::vector<std::variant<ParallelSplit, Node>> children =
        vector_of(s.children);
    if (children.size() == 1) {
      return from_series_child(get_only(children));
    } else if (children.size() == 2) {
      return BinarySPDecompositionTree{BinarySeriesSplit{
          from_series_child(children[0]), from_series_child(children[1])}};
    }

    auto s1 = slice(children, 0, children.size() / 2);
    auto s2 = slice(children, children.size() / 2, std::nullopt);

    return BinarySPDecompositionTree{BinarySeriesSplit{
        from_series(SeriesSplit{s1}), from_series(SeriesSplit{s2})}};
  };

  from_parallel_child = [&](std::variant<SeriesSplit, Node> const &v)
      -> BinarySPDecompositionTree {
    return std::visit(overload{
                          [&](Node const &n) { return from_node(n); },
                          [&](SeriesSplit const &s) { return from_series(s); },
                      },
                      v);
  };

  from_series_child = [&](std::variant<ParallelSplit, Node> const &v)
      -> BinarySPDecompositionTree {
    return std::visit(
        overload{
            [&](Node const &n) { return from_node(n); },
            [&](ParallelSplit const &p) { return from_parallel(p); },
        },
        v);
  };

  return nary.visit<BinarySPDecompositionTree>(overload{
      [&](Node const &n) { return from_node(n); },
      [&](SeriesSplit const &s) { return from_series(s); },
      [&](ParallelSplit const &p) { return from_parallel(p); },
  });
}

} // namespace FlexFlow
