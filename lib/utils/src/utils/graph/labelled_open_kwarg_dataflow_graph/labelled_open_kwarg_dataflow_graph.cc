#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

template struct LabelledOpenKwargDataflowGraph<value_type<0>,
                                               value_type<1>,
                                               ordered_value_type<2>,
                                               ordered_value_type<3>>;

template struct LabelledOpenKwargDataflowGraph<ordered_value_type<0>,
                                               ordered_value_type<0>,
                                               ordered_value_type<0>,
                                               ordered_value_type<0>>;

} // namespace FlexFlow
