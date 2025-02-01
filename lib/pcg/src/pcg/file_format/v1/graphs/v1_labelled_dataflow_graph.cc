#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using OutputLabel = value_type<1>;

template std::pair<V1LabelledDataflowGraph<NodeLabel, OutputLabel>,
                   bidict<nonnegative_int, Node>>
    to_v1_including_node_numbering(
        LabelledDataflowGraphView<NodeLabel, OutputLabel> const &);

template V1LabelledDataflowGraph<NodeLabel, OutputLabel>
    to_v1(LabelledDataflowGraphView<NodeLabel, OutputLabel> const &);

} // namespace FlexFlow
