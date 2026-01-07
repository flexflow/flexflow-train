#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

template struct LabelledKwargDataflowGraph<value_type<0>,
                                           value_type<1>,
                                           ordered_value_type<2>>;

template struct LabelledKwargDataflowGraph<ordered_value_type<0>,
                                           ordered_value_type<0>,
                                           ordered_value_type<0>>;

} // namespace FlexFlow
