#include "utils/graph/series_parallel/non_normal_sp_decomposition.h"
#include "utils/containers/all_of.h"
#include "utils/containers/extend.h"
#include "utils/containers/multiset_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/series_parallel/non_normal_parallel_split.dtg.h"
#include "utils/graph/series_parallel/non_normal_series_split.dtg.h"
#include "utils/graph/series_parallel/parallel_split.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/graph/series_parallel/series_split.dtg.h"
#include "utils/overload.h"
#include "utils/variant.h"

namespace FlexFlow {

NonNormalSPDecomposition non_normal_series_composition(
    std::vector<NonNormalSPDecomposition> const &sp_compositions) {

  std::vector<std::variant<NonNormalParallelSplit, Node>> composition{};

  for (NonNormalSPDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<NonNormalSeriesSplit>()) {
      extend(composition, sp_comp.get<NonNormalSeriesSplit>().children);
    } else if (sp_comp.has<NonNormalParallelSplit>()) {
      composition.push_back(sp_comp.get<NonNormalParallelSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.push_back(sp_comp.get<Node>());
    }
  }

  return NonNormalSPDecomposition{NonNormalSeriesSplit{composition}};
}

NonNormalSPDecomposition non_normal_parallel_composition(
    std::unordered_multiset<NonNormalSPDecomposition> const &sp_compositions) {

  std::unordered_multiset<
      std::variant<::FlexFlow::NonNormalSeriesSplit, ::FlexFlow::Node>>
      composition{};

  for (NonNormalSPDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<NonNormalParallelSplit>()) {
      composition = multiset_union(
          composition, sp_comp.get<NonNormalParallelSplit>().get_children());
    } else if (sp_comp.has<NonNormalSeriesSplit>()) {
      composition.insert(sp_comp.get<NonNormalSeriesSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.insert(sp_comp.get<Node>());
    }
  }
  return NonNormalSPDecomposition(NonNormalParallelSplit{composition});
}

static Node as_non_normal(Node const &n) { return n; }

static NonNormalSeriesSplit as_non_normal(SeriesSplit const &s) {
  return non_normal_series_composition(
             transform(s.children,
                       [](std::variant<ParallelSplit, Node> const &child) {
                         return as_non_normal(
                             widen<SeriesParallelDecomposition>(child));
                       }))
      .get<NonNormalSeriesSplit>();
}

static NonNormalParallelSplit as_non_normal(ParallelSplit const &p) {
  return non_normal_parallel_composition(
             transform(p.get_children(),
                       [](std::variant<SeriesSplit, Node> const &child) {
                         return as_non_normal(
                             widen<SeriesParallelDecomposition>(child));
                       }))
      .get<NonNormalParallelSplit>();
}

NonNormalSPDecomposition as_non_normal(SeriesParallelDecomposition const &sp) {
  return sp.visit<NonNormalSPDecomposition>(
      [](auto const &t) { return NonNormalSPDecomposition{as_non_normal(t)}; });
}

bool is_empty_non_normal(NonNormalSPDecomposition const &sp) {
  return sp.visit<bool>(overload{
      [](Node const &) { return false; },
      [](NonNormalSeriesSplit const &serial) {
        return all_of(serial.children, [](auto const &child) {
          return is_empty_non_normal(widen<NonNormalSPDecomposition>(child));
        });
      },
      [](NonNormalParallelSplit const &parallel) {
        return all_of(parallel.get_children(), [](auto const &child) {
          return is_empty_non_normal(widen<NonNormalSPDecomposition>(child));
        });
      },
  });
}

} // namespace FlexFlow
