#include "substitutions/sub_parallel_computation_graph_data.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/transform.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.h"

namespace FlexFlow {

void require_sub_parallel_computation_graph_data_is_valid(
    SubParallelComputationGraphData const &d) {
  LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                     ParallelTensorAttrs,
                                     int,
                                     TensorSlotName>
      labelled_graph_data =
          LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                             ParallelTensorAttrs,
                                             int,
                                             TensorSlotName>{
              /*node_data=*/map_keys(d.node_data,
                                     [](parallel_layer_guid_t l) -> Node {
                                       return l.raw_graph_node;
                                     }),
              /*edges=*/
              transform(d.edges,
                        [](SubParallelComputationGraphEdge const &e)
                            -> OpenKwargDataflowEdge<int, TensorSlotName> {
                          return e.raw_edge;
                        }),
              /*inputs=*/
              transform(d.inputs,
                        [](input_parallel_tensor_guid_t const &i)
                            -> KwargDataflowGraphInput<int> {
                          return i.raw_dataflow_graph_input;
                        }),
              /*value_data=*/
              map_keys(d.value_data,
                       [](open_parallel_tensor_guid_t t)
                           -> OpenKwargDataflowValue<int, TensorSlotName> {
                         return t.raw_open_dataflow_value;
                       }),
          };

  require_labelled_open_kwarg_dataflow_graph_data_is_valid(labelled_graph_data);
}

} // namespace FlexFlow
