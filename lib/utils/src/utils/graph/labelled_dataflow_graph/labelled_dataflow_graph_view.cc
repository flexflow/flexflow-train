#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using OutputLabel = value_type<1>;

template struct LabelledDataflowGraphView<NodeLabel, OutputLabel>;

} // namespace FlexFlow
