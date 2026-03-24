#include "pcg/file_format/v1/v1_parallel_computation_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"

namespace FlexFlow {

V1ParallelComputationGraph to_v1(ParallelComputationGraph const &g) {
  return V1ParallelComputationGraph{
      to_v1<ParallelLayerAttrs, ParallelTensorAttrs, TensorSlotName>(
          g.raw_graph),
  };
}

ParallelComputationGraph from_v1(V1ParallelComputationGraph const &v1) {
  LabelledKwargDataflowGraph<ParallelLayerAttrs,
                             ParallelTensorAttrs,
                             TensorSlotName>
      raw_graph = LabelledKwargDataflowGraph<ParallelLayerAttrs,
                                             ParallelTensorAttrs,
                                             TensorSlotName>::
          create_copy_of<LabelledKwargDataflowGraph<ParallelLayerAttrs,
                                                    ParallelTensorAttrs,
                                                    TensorSlotName>>(
              from_v1(v1.raw_graph).first);
  return ParallelComputationGraph{raw_graph};
}

} // namespace FlexFlow
