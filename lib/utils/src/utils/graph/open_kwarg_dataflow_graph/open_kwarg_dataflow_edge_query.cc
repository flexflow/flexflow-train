#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template
  OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>
    open_kwarg_dataflow_edge_query_all();

template
  bool open_kwarg_dataflow_edge_query_includes(
    OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &,
    OpenKwargDataflowEdge<GraphInputName, SlotName> const &);


} // namespace FlexFlow
