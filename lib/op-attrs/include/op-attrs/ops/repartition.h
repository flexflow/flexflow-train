#ifndef _FLEXFLOW_PARTITION_ATTRS_H
#define _FLEXFLOW_PARTITION_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "utils/record_formatter.h"
#include <tl/expected.hpp>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(RepartitionAttrs);

RecordFormatter as_dot(RepartitionAttrs const &);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(RepartitionAttrs const &,
                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
