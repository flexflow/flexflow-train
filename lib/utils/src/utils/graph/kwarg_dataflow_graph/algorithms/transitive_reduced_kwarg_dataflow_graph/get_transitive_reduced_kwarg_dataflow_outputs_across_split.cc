#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/get_transitive_reduced_kwarg_dataflow_outputs_across_split.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template std::unordered_set<KwargDataflowOutput<SlotName>>
    get_transitive_reduced_kwarg_dataflow_outputs_across_split(
        TransitiveReducedKwargDataflowGraphView<SlotName> const &,
        BinarySeriesSplit const &);

} // namespace FlexFlow
