#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/materialize_labelled_kwarg_dataflow_graph_view.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using SlotName = ordered_value_type<2>;

template LabelledKwargDataflowGraph<NodeLabel, ValueLabel, SlotName>
    materialize_labelled_kwarg_dataflow_graph_view(
        LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const
            &);

} // namespace FlexFlow
