#include "utils/graph/series_parallel/get_ancestors.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/variant.h"
#include <cassert>

namespace FlexFlow {

static bool perform_traversal(SeriesParallelDecomposition const &sp,
                              Node const &starting_node,
                              std::unordered_set<Node> &ancestors) {
  return sp.visit<bool>([&](auto const &sp) {
    return perform_traversal(sp, starting_node, ancestors);
  });
}

static bool perform_traversal(SeriesSplit const &serial,
                              Node const &starting_node,
                              std::unordered_set<Node> &ancestors) {
  std::vector<SeriesParallelDecomposition> children =
      transform(serial.children, [](auto const &child) {
        return widen<SeriesParallelDecomposition>(child);
      });
  for (SeriesParallelDecomposition const &child : children) {
    bool found_starting_node =
        perform_traversal(child, starting_node, ancestors);
    if (found_starting_node) {
      return true;
    }
  }
  return false;
}

static bool perform_traversal(ParallelSplit const &parallel,
                              Node const &starting_node,
                              std::unordered_set<Node> &ancestors) {
  std::unordered_multiset<SeriesParallelDecomposition> children =
      transform(parallel.get_children(), [](auto const &child) {
        return widen<SeriesParallelDecomposition>(child);
      });

  if (contains(get_nodes(parallel), starting_node)) {
    SeriesParallelDecomposition branch_with_starting_node = get_only(
        filter(children, [&](SeriesParallelDecomposition const &child) {
          return contains(get_nodes(child), starting_node);
        }));
    perform_traversal(branch_with_starting_node, starting_node, ancestors);
    return true;
  }

  for (SeriesParallelDecomposition const &child : children) {
    perform_traversal(child, starting_node, ancestors);
  }
  return false;
}

static bool perform_traversal(Node const &node,
                              Node const &starting_node,
                              std::unordered_set<Node> &ancestors) {
  if (starting_node != node) {
    ancestors.insert(node);
    return false;
  }
  return true;
}

std::unordered_set<Node> get_ancestors(SeriesParallelDecomposition const &sp,
                                       Node const &starting_node) {
  assert(contains(get_nodes(sp), starting_node));
  std::unordered_set<Node> ancestors;
  perform_traversal(sp, starting_node, ancestors);
  return ancestors;
}

} // namespace FlexFlow
