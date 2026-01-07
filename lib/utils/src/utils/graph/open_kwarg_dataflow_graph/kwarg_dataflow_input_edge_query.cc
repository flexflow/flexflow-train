#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_input_edge_query.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template KwargDataflowInputEdgeQuery<GraphInputName, SlotName>
    kwarg_dataflow_input_edge_query_all();

template KwargDataflowInputEdgeQuery<GraphInputName, SlotName>
    kwarg_dataflow_input_edge_query_none();

template bool kwarg_dataflow_input_edge_query_includes(
    KwargDataflowInputEdgeQuery<GraphInputName, SlotName> const &,
    KwargDataflowInputEdge<GraphInputName, SlotName> const &);

} // namespace FlexFlow
