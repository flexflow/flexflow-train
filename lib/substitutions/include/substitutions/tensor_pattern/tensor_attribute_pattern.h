#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_TENSOR_ATTRIBUTE_PATTERN_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_TENSOR_ATTRIBUTE_PATTERN_H

#include "substitutions/tensor_pattern/tensor_attribute_pattern.dtg.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

TensorAttributePattern tensor_attribute_pattern_match_all();
TensorAttributePattern
    tensor_attr_pattern_require_num_dims(positive_int num_dims);

} // namespace FlexFlow

#endif
