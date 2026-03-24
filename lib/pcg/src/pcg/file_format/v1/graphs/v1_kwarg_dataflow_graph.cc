#include "pcg/file_format/v1/graphs/v1_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template V1KwargDataflowGraph<SlotName>
    to_v1(KwargDataflowGraphView<SlotName> const &);

template V1KwargDataflowGraph<SlotName>
    to_v1(KwargDataflowGraphView<SlotName> const &,
          std::unordered_map<Node, nonnegative_int> const &);

template std::pair<KwargDataflowGraphView<SlotName>,
                   std::unordered_map<nonnegative_int, Node>>
    from_v1_including_node_numbering(V1KwargDataflowGraph<SlotName> const &);

template KwargDataflowGraphView<SlotName>
    from_v1(V1KwargDataflowGraph<SlotName> const &);

} // namespace FlexFlow
