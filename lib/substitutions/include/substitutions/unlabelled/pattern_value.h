#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_VALUE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_VALUE_H

#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"

namespace FlexFlow {

OpenKwargDataflowValue<int, TensorSlotName>
    raw_open_dataflow_value_from_pattern_value(PatternValue const &);
PatternValue
    pattern_value_from_raw_open_kwarg_dataflow_value(OpenKwargDataflowValue<int, TensorSlotName> const &);

} // namespace FlexFlow

#endif
