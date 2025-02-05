#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_LAYER_ATTRS_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_LAYER_ATTRS_H

#include "pcg/layer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"

namespace FlexFlow {

OperatorType get_op_type(ParallelLayerAttrs const &);

ParallelLayerAttrs parallel_layer_attrs_from_layer_attrs(LayerAttrs const &);

} // namespace FlexFlow

#endif
