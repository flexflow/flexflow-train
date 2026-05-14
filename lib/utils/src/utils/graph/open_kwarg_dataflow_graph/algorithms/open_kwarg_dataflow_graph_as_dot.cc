#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_as_dot.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template std::string open_kwarg_dataflow_graph_as_dot(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
    std::function<nlohmann::json(Node const &)> const &,
    std::function<nlohmann::json(
        OpenKwargDataflowValue<GraphInputName, SlotName> const &)> const &,
    std::function<nlohmann::json(SlotName const &)> const &,
    std::function<
        std::vector<SlotName>(std::unordered_set<SlotName> const &)> const &);

} // namespace FlexFlow
