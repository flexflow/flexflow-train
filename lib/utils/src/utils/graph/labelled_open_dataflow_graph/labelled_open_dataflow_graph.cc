#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;

template struct LabelledOpenDataflowGraph<NodeLabel, ValueLabel>;

} // namespace FlexFlow
