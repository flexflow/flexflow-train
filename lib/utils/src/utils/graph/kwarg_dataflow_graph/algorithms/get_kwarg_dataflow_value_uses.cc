#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_value_uses.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template
  std::unordered_set<KwargDataflowInput<SlotName>>
      get_kwarg_dataflow_value_uses(
          KwargDataflowGraphView<SlotName> const &,
          KwargDataflowOutput<SlotName> const &);

} // namespace FlexFlow
