#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H

#include "utils/containers/get_all_permutations.h"
#include "utils/containers/zip.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/is_isomorphic_under.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/algorithms/find_isomorphisms.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"

namespace FlexFlow {

/**
 * @brief Finds an isomorphism between \p src and \p dst, if one exists.
 *
 * @note If multiple isomorphisms exist, an arbitrary one is returned.
 */
template <typename NodeLabel, typename ValueLabel>
std::optional<OpenDataflowGraphIsomorphism> find_isomorphism(
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &src,
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &dst) {
  std::unordered_set<OpenDataflowGraphIsomorphism> unlabelled_isomorphisms =
      find_isomorphisms(static_cast<OpenDataflowGraphView>(src),
                        static_cast<OpenDataflowGraphView>(dst));

  for (OpenDataflowGraphIsomorphism const &candidate_isomorphism :
       unlabelled_isomorphisms) {
    if (is_isomorphic_under(src, dst, candidate_isomorphism)) {
      return candidate_isomorphism;
    }
  }

  return std::nullopt;
}

} // namespace FlexFlow

#endif
