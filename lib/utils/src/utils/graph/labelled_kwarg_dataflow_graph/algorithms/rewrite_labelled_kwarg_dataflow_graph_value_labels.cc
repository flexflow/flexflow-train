#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/rewrite_labelled_kwarg_dataflow_graph_value_labels.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using SlotName = value_type<2>;
using NewValueLabel = value_type<3>;
using F = std::function<NewValueLabel(KwargDataflowOutput<SlotName> const &,
                                      ValueLabel const &)>;

LabelledKwargDataflowGraphView<NodeLabel, NewValueLabel, SlotName>
    rewrite_labelled_kwarg_dataflow_graph_value_labels(
        LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const &,
        F);

} // namespace FlexFlow
