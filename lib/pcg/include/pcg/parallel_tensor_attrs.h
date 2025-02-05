#ifndef _PCG_PARALLEL_TENSOR_ATTRS_H
#define _PCG_PARALLEL_TENSOR_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "pcg/tensor_attrs.dtg.h"

namespace FlexFlow {

TensorAttrs get_piece_attrs(ParallelTensorAttrs const &);

ParallelTensorAttrs
    parallel_tensor_attrs_from_tensor_attrs(TensorAttrs const &);

} // namespace FlexFlow

#endif
