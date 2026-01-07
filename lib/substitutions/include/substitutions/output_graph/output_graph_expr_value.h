#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_GRAPH_EXPR_VALUE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_GRAPH_EXPR_VALUE_H

#include "substitutions/output_graph/output_graph_expr_value.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"

namespace FlexFlow {

OpenKwargDataflowValue<int, TensorSlotName>
    raw_open_kwarg_dataflow_value_from_output_graph_expr_value(
        OutputGraphExprValue const &);
OutputGraphExprValue output_graph_expr_value_from_raw_open_kwarg_dataflow_value(
    OpenKwargDataflowValue<int, TensorSlotName> const &);

} // namespace FlexFlow

#endif
