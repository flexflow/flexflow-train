#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_kwarg_dataflow_graph_view_as_dot.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using SlotName = ordered_value_type<2>;

template std::string labelled_kwarg_dataflow_graph_view_as_dot(
    LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const &,
    std::function<nlohmann::json(NodeLabel const &)> const &,
    std::function<nlohmann::json(ValueLabel const &)> const &,
    std::function<nlohmann::json(SlotName const &)> const &,
    std::function<
        std::vector<SlotName>(std::unordered_set<SlotName> const &)> const &);

} // namespace FlexFlow
