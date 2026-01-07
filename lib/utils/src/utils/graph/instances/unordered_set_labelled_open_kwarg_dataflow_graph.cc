#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

template struct UnorderedSetLabelledOpenKwargDataflowGraph<
    value_type<0>,
    value_type<1>,
    ordered_value_type<2>,
    ordered_value_type<3>>;

template struct UnorderedSetLabelledOpenKwargDataflowGraph<
    ordered_value_type<0>,
    ordered_value_type<0>,
    ordered_value_type<0>,
    ordered_value_type<0>>;

} // namespace FlexFlow
