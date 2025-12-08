#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template
  std::unordered_set<OpenKwargDataflowValue<GraphInputName, SlotName>> 
    get_all_open_kwarg_dataflow_values(
      OpenKwargDataflowGraphView<GraphInputName, SlotName> const &);

} // namespace FlexFlow
