#include "utils/graph/series_parallel/get_ancestors.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/variant.h"
#include <cassert>

namespace FlexFlow {

std::unordered_set<Node> get_ancestors(SeriesParallelDecomposition const &sp,
                                       Node const &node);

static std::unordered_set<Node> get_ancestors(Node const &, Node const &node) {
  return {};
}

static std::unordered_set<Node> get_ancestors(SeriesSplit const &serial,
                                              Node const &node) {
  std::unordered_set<Node> ancestors{};
  for (std::variant<ParallelSplit, Node> const &child : serial.children) {
    SeriesParallelDecomposition child_sp =
        widen<SeriesParallelDecomposition>(child);
    if (contains(get_nodes(child_sp), node)) {
      return set_union(ancestors, get_ancestors(child_sp, node));
    }
    ancestors = set_union(ancestors, unordered_set_of(get_nodes(child_sp)));
  }
  throw std::runtime_error("node not found in SeriesSplit");
}

static std::unordered_set<Node> get_ancestors(ParallelSplit const &parallel,
                                              Node const &node) {
  SeriesParallelDecomposition branch =
      get_only(filter(transform(parallel.get_children(),
                                [](std::variant<SeriesSplit, Node> const &c) {
                                  return widen<SeriesParallelDecomposition>(c);
                                }),
                      [&](SeriesParallelDecomposition const &child) {
                        return contains(get_nodes(child), node);
                      }));
  return get_ancestors(branch, node);
}

std::unordered_set<Node> get_ancestors(SeriesParallelDecomposition const &sp,
                                       Node const &node) {
  assert(contains(get_nodes(sp), node));
  return sp.visit<std::unordered_set<Node>>(
      [&](auto const &t) { return get_ancestors(t, node); });
}

} // namespace FlexFlow
