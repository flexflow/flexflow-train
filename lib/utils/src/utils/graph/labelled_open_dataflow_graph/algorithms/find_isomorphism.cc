#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;

template std::optional<OpenDataflowGraphIsomorphism> find_isomorphism(
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &,
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &);

} // namespace FlexFlow
