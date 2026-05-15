#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_LAYER_ATTRS_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_LAYER_ATTRS_H

#include "pcg/mapped_parallel_computation_graph/mapped_parallel_layer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"

namespace FlexFlow {

ParallelLayerAttrs
    unmapped_parallel_layer_attrs_from_mapped(MappedParallelLayerAttrs const &);

MappedParallelLayerAttrs mapped_parallel_layer_attrs_without_layer_name(
    MappedParallelLayerAttrs const &);

} // namespace FlexFlow

#endif
