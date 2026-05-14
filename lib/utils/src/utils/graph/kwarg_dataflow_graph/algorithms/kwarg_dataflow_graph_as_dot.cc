#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_as_dot.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template std::string kwarg_dataflow_graph_as_dot(
    KwargDataflowGraphView<SlotName> const &,
    std::function<nlohmann::json(Node const &)> const &,
    std::function<nlohmann::json(KwargDataflowOutput<SlotName> const &)> const
        &,
    std::function<nlohmann::json(SlotName const &)> const &,
    std::function<
        std::vector<SlotName>(std::unordered_set<SlotName> const &)> const &);

} // namespace FlexFlow
