#include "utils/graph/open_kwarg_dataflow_graph/algorithms/permute_open_kwarg_dataflow_graph_input_ids.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template
  OpenKwargDataflowGraphView<GraphInputName, SlotName>
    permute_open_kwarg_dataflow_graph_input_ids(
      OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
      bidict<KwargDataflowGraphInput<GraphInputName>, KwargDataflowGraphInput<GraphInputName>> const &);


} // namespace FlexFlow
