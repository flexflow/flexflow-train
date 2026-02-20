#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template KwargDataflowEdgeQuery<SlotName> kwarg_dataflow_edge_query_all();

template KwargDataflowEdgeQuery<SlotName> kwarg_dataflow_edge_query_none();

template bool
    kwarg_dataflow_edge_query_includes(KwargDataflowEdgeQuery<SlotName> const &,
                                       KwargDataflowEdge<SlotName> const &);

} // namespace FlexFlow
