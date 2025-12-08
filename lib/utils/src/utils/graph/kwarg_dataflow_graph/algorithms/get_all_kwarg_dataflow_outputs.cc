#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template
  std::unordered_set<KwargDataflowOutput<SlotName>>
    get_all_kwarg_dataflow_outputs(
        KwargDataflowGraphView<SlotName> const &);

} // namespace FlexFlow
