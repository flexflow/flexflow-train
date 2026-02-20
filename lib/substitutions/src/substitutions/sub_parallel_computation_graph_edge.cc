#include "substitutions/sub_parallel_computation_graph_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"

namespace FlexFlow {

SubParallelComputationGraphEdge
    subpcg_edge_from_tensor_and_dst(parallel_tensor_guid_t const &tensor,
                                    parallel_layer_guid_t const &layer,
                                    TensorSlotName input_slot_name) {
  return SubParallelComputationGraphEdge{
      OpenKwargDataflowEdge<int, TensorSlotName>{
          KwargDataflowEdge<TensorSlotName>{
              tensor.raw_graph_output,
              KwargDataflowInput<TensorSlotName>{
                  layer.raw_graph_node,
                  input_slot_name,
              },
          },
      },
  };
}

SubParallelComputationGraphEdge
    subpcg_edge_from_tensor_and_use(open_parallel_tensor_guid_t const &tensor,
                                    parallel_tensor_use_t const &use) {
  return SubParallelComputationGraphEdge{
      mk_open_kwarg_dataflow_edge_from_src_val_and_dst(
          tensor.raw_open_dataflow_value, use.raw_dataflow_input),
  };
}

open_parallel_tensor_guid_t
    get_parallel_tensor(SubParallelComputationGraphEdge const &e) {
  OpenKwargDataflowValue<int, TensorSlotName> raw_value =
      get_src_of_open_kwarg_dataflow_edge(e.raw_edge);
  return open_parallel_tensor_guid_t{raw_value};
}

} // namespace FlexFlow
