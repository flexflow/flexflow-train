#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template KwargDataflowOutputQuery<SlotName> kwarg_dataflow_output_query_all();

template bool kwarg_dataflow_output_query_includes(
    KwargDataflowOutputQuery<SlotName> const &,
    KwargDataflowOutput<SlotName> const &);

} // namespace FlexFlow
