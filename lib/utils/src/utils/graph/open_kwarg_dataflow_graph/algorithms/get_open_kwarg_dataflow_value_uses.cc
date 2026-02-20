#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_value_uses.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template std::unordered_set<KwargDataflowInput<SlotName>>
    get_open_kwarg_dataflow_value_uses(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
        OpenKwargDataflowValue<GraphInputName, SlotName> const &);

} // namespace FlexFlow
