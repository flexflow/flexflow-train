#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using NewValueLabel = value_type<2>;
using F =
    std::function<NewValueLabel(OpenDataflowValue const &, ValueLabel const &)>;

template LabelledDataflowGraphView<NodeLabel, NewValueLabel>
    rewrite_value_labels(
        LabelledDataflowGraphView<NodeLabel, ValueLabel> const &, F);

} // namespace FlexFlow
