#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;
using NewValueLabel = value_type<4>;
using F = std::function<NewValueLabel(OpenKwargDataflowValue<GraphInputName, SlotName> const &, ValueLabel const &)>;

LabelledOpenKwargDataflowGraphView<NodeLabel, NewValueLabel, GraphInputName, SlotName> rewrite_value_labels(
    LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &, F);

} // namespace FlexFlow
