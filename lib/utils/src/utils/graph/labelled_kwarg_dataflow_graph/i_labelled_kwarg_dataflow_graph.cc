#include "utils/graph/labelled_kwarg_dataflow_graph/i_labelled_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

template struct ILabelledKwargDataflowGraph<value_type<0>,
                                            value_type<1>,
                                            ordered_value_type<2>>;

template struct ILabelledKwargDataflowGraph<ordered_value_type<0>,
                                            ordered_value_type<0>,
                                            ordered_value_type<0>>;

} // namespace FlexFlow
