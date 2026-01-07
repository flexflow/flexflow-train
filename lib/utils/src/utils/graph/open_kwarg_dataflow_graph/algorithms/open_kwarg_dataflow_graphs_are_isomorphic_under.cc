#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graphs_are_isomorphic_under.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

bool open_kwarg_dataflow_graphs_are_isomorphic_under(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
    OpenKwargDataflowGraphIsomorphism<GraphInputName> const &);

} // namespace FlexFlow
