#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template 
  OpenKwargDataflowGraphView<GraphInputName, SlotName>
    view_from_open_kwarg_dataflow_graph_data(
      OpenKwargDataflowGraphData<GraphInputName, SlotName> const &);

} // namespace FlexFlow
