#include "utils/graph/open_kwarg_dataflow_graph/i_open_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

template struct IOpenKwargDataflowGraph<ordered_value_type<0>,
                                        ordered_value_type<1>>;
template struct IOpenKwargDataflowGraph<ordered_value_type<0>,
                                        ordered_value_type<0>>;

} // namespace FlexFlow
