#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
  get_all_kwarg_dataflow_graph_inputs(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &);

} // namespace FlexFlow
