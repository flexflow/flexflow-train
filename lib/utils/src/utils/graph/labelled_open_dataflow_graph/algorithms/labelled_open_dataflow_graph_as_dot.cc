#include "utils/graph/labelled_open_dataflow_graph/algorithms/labelled_open_dataflow_graph_as_dot.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;

template std::string labelled_open_dataflow_graph_as_dot(
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &,
    std::function<std::string(NodeLabel const &)> const &,
    std::function<std::string(ValueLabel const &)> const &,
    std::function<std::string(DataflowGraphInput const &)> const &,
    std::function<std::string(DataflowInput const &)> const &,
    std::function<std::string(DataflowOutput const &)> const &);

} // namespace FlexFlow
