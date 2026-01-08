#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graphs_are_isomorphic_under.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;

template bool labelled_open_kwarg_dataflow_graphs_are_isomorphic_under(
    LabelledOpenKwargDataflowGraphView<NodeLabel,
                                       ValueLabel,
                                       GraphInputName,
                                       SlotName> const &,
    LabelledOpenKwargDataflowGraphView<NodeLabel,
                                       ValueLabel,
                                       GraphInputName,
                                       SlotName> const &,
    OpenKwargDataflowGraphIsomorphism<GraphInputName> const &);

} // namespace FlexFlow
