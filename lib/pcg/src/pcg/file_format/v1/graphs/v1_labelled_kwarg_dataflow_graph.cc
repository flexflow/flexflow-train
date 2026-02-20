#include "pcg/file_format/v1/graphs/v1_labelled_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using OutputLabel = value_type<1>;
using SlotName = ordered_value_type<2>;

template std::pair<
    V1LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>,
    bidict<nonnegative_int, Node>>
    to_v1_including_node_numbering(
        LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const
            &);

template V1LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName> to_v1(
    LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const &);

} // namespace FlexFlow
