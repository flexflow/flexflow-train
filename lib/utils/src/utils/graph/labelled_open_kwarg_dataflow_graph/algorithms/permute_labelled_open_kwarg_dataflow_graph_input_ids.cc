#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/permute_labelled_open_kwarg_dataflow_graph_input_ids.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;

template
  LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> 
    permute_labelled_open_kwarg_dataflow_graph_input_ids(
      LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &,
      bidict<KwargDataflowGraphInput<GraphInputName>, KwargDataflowGraphInput<GraphInputName>> const &);

} // namespace FlexFlow
