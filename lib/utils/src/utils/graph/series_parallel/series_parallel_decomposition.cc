#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/containers/all_of.h"
#include "utils/containers/extend.h"
#include "utils/containers/multiset_union.h"
#include "utils/containers/set_union.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/series_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/hash/unordered_set.h"
#include "utils/variant.h"
#include <unordered_set>

namespace FlexFlow {

struct ToFinalAST {
  std::variant<SeriesSplit, ParallelSplit, Node>
      operator()(IntermediateSpDecompositionTree const &node) {
    if (node.type == SplitType::SERIES) {
      return SeriesSplit{transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<ParallelSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          })};
    } else {
      return ParallelSplit{unordered_multiset_of(transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<SeriesSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          }))};
    }
  }

  std::variant<SeriesSplit, ParallelSplit, Node> operator()(Node const &node) {
    return node;
  }
};

std::variant<SeriesSplit, ParallelSplit, Node> internal_to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit(ToFinalAST{}, flatten_ast(ast));
}

SeriesParallelDecomposition to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit([](auto &&x) { return SeriesParallelDecomposition{x}; },
                    internal_to_final_ast(ast));
}

std::unordered_multiset<Node> get_nodes(SeriesParallelDecomposition const &sp) {
  return sp.visit<std::unordered_multiset<Node>>(
      [](auto &&t) { return get_nodes(t); });
}

std::unordered_multiset<Node> get_nodes(SeriesSplit const &serial) {
  return multiset_union(transform(
      serial.children,
      [](std::variant<ParallelSplit, Node> const &child)
          -> std::unordered_multiset<Node> {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_multiset<Node> get_nodes(ParallelSplit const &parallel) {
  return multiset_union(transform(
      vector_of(parallel.get_children()),
      [](std::variant<SeriesSplit, Node> const &child) {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_multiset<Node> get_nodes(Node const &node) {
  return {node};
}

bool is_empty(Node const &node) {
  return false;
}

bool is_empty(SeriesSplit const &serial) {
  return all_of(serial.children, [](auto const &child) {
    return is_empty(widen<SeriesParallelDecomposition>(child));
  });
}

bool is_empty(ParallelSplit const &parallel) {
  return all_of(parallel.get_children(), [](auto const &child) {
    return is_empty(widen<SeriesParallelDecomposition>(child));
  });
}

bool is_empty(SeriesParallelDecomposition const &sp) {
  return sp.visit<bool>([](auto const &t) { return is_empty(t); });
}

SeriesParallelDecomposition series_composition(
    std::vector<SeriesParallelDecomposition> const &sp_compositions) {
  std::vector<std::variant<ParallelSplit, Node>> composition{};
  for (SeriesParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<SeriesSplit>()) {
      extend(composition, sp_comp.get<SeriesSplit>().children);
    } else if (sp_comp.has<ParallelSplit>()) {
      composition.push_back(sp_comp.get<ParallelSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.push_back(sp_comp.get<Node>());
    }
  }
  return SeriesParallelDecomposition{SeriesSplit{composition}};
}

SeriesParallelDecomposition parallel_composition(
    std::unordered_multiset<SeriesParallelDecomposition> const
        &sp_compositions) {
  std::unordered_multiset<
      std::variant<::FlexFlow::SeriesSplit, ::FlexFlow::Node>>
      composition{};
  for (SeriesParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<ParallelSplit>()) {
      composition = multiset_union(composition,
                                   sp_comp.get<ParallelSplit>().get_children());
    } else if (sp_comp.has<SeriesSplit>()) {
      composition.insert(sp_comp.get<SeriesSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.insert(sp_comp.get<Node>());
    }
  }
  return SeriesParallelDecomposition(ParallelSplit{composition});
}

} // namespace FlexFlow
