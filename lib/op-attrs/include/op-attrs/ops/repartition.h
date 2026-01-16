#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_H

#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "utils/record_formatter.h"
#include <tl/expected.hpp>

namespace FlexFlow {

RecordFormatter as_dot(RepartitionAttrs const &);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(RepartitionAttrs const &,
                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
