#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_view_as_dot.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;

template std::string labelled_open_kwarg_dataflow_graph_view_as_dot(
    LabelledOpenKwargDataflowGraphView<NodeLabel,
                                       ValueLabel,
                                       GraphInputName,
                                       SlotName> const &g,
    std::function<std::string(NodeLabel const &)> const &,
    std::function<std::string(ValueLabel const &)> const &);

} // namespace FlexFlow
