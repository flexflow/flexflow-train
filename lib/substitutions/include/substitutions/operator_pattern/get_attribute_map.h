#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_GET_ATTRIBUTE_MAP_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_GET_ATTRIBUTE_MAP_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_key.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"

namespace FlexFlow {

std::unordered_map<OperatorAttributeKey, OperatorAttributeValue>
    get_attribute_map(PCGOperatorAttrs const &);

} // namespace FlexFlow

#endif
