#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using OutputLabel = value_type<1>;

template struct LabelledDataflowGraph<NodeLabel, OutputLabel>;

} // namespace FlexFlow
