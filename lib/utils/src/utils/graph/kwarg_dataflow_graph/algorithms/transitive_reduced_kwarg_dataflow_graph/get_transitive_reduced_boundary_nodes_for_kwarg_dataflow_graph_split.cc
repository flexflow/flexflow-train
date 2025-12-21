#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/get_transitive_reduced_boundary_nodes_for_kwarg_dataflow_graph_split.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template
  SplitBoundaryNodes get_transitive_reduced_boundary_nodes_for_kwarg_dataflow_graph_split(
      TransitiveReducedKwargDataflowGraphView<SlotName> const &, BinarySeriesSplit const &);

} // namespace FlexFlow
