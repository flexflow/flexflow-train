#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template 
  std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> get_outgoing_kwarg_dataflow_outputs_for_node(
    KwargDataflowGraphView<SlotName> const &,
    Node const &);

} // namespace FlexFlow
