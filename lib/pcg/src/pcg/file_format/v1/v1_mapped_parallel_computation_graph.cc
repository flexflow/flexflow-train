#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_kwarg_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"

namespace FlexFlow {

V1MappedParallelComputationGraph
    to_v1(MappedParallelComputationGraph const &mpcg) {
  return V1MappedParallelComputationGraph{
      to_v1<MappedParallelLayerAttrs, ParallelTensorAttrs, TensorSlotName>(
          mpcg.raw_graph),
  };
}

MappedParallelComputationGraph
    from_v1(V1MappedParallelComputationGraph const &v1) {
  return MappedParallelComputationGraph{
      LabelledKwargDataflowGraph<MappedParallelLayerAttrs,
                                 ParallelTensorAttrs,
                                 TensorSlotName>::
          create_copy_of<UnorderedSetLabelledOpenKwargDataflowGraph<
              MappedParallelLayerAttrs,
              ParallelTensorAttrs,
              int,
              TensorSlotName>>(from_v1(v1.raw_graph))};
}

} // namespace FlexFlow
