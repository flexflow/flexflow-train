#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using GraphInputName = value_type<0>;
using SlotName = value_type<1>;

template
  OpenKwargDataflowEdge<GraphInputName, SlotName>
    mk_open_kwarg_dataflow_edge_from_src_val_and_dst(
      OpenKwargDataflowValue<GraphInputName, SlotName> const &,
      KwargDataflowInput<SlotName> const &);

} // namespace FlexFlow
