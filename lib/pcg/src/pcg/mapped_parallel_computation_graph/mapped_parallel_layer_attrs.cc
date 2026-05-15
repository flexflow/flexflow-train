#include "pcg/mapped_parallel_computation_graph/mapped_parallel_layer_attrs.h"

namespace FlexFlow {

ParallelLayerAttrs unmapped_parallel_layer_attrs_from_mapped(
    MappedParallelLayerAttrs const &mapped) {
  return ParallelLayerAttrs{
      /*op_attrs=*/mapped.op_attrs,
      /*name=*/mapped.name,
  };
}

MappedParallelLayerAttrs mapped_parallel_layer_attrs_without_layer_name(
    MappedParallelLayerAttrs const &m) {
  MappedParallelLayerAttrs result = m;
  result.name = std::nullopt;
  return result;
}

} // namespace FlexFlow
