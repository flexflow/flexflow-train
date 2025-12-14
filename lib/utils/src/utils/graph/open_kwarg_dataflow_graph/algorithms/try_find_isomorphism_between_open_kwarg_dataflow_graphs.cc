#include "utils/graph/open_kwarg_dataflow_graph/algorithms/try_find_isomorphism_between_open_kwarg_dataflow_graphs.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template
  std::optional<OpenKwargDataflowGraphIsomorphism<GraphInputName>>
      try_find_isomorphism_between_open_kwarg_dataflow_graphs(
          OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
          OpenKwargDataflowGraphView<GraphInputName, SlotName> const &);

} // namespace FlexFlow
