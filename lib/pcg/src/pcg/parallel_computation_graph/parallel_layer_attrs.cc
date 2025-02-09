#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"

namespace FlexFlow {

OperatorType get_op_type(ParallelLayerAttrs const &a) {
  return get_op_type(a.op_attrs);
}

ParallelLayerAttrs
    parallel_layer_attrs_from_layer_attrs(LayerAttrs const &layer_attrs) {
  return ParallelLayerAttrs{
      pcg_op_attrs_from_compgraph_op_attrs(layer_attrs.attrs),
      layer_attrs.name};
}

} // namespace FlexFlow
