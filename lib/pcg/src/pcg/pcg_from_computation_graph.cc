#include "pcg/pcg_from_computation_graph.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "pcg/parallel_tensor_attrs.h"
#include "pcg/tensor_attrs.dtg.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

ParallelComputationGraph
    pcg_from_computation_graph(ComputationGraph const &cg) {
  auto layer_map = [&](Node const &_, LayerAttrs const &layer) {
    return parallel_layer_attrs_from_layer_attrs(layer);
  };
  auto tensor_map = [&](OpenDataflowValue const &_, TensorAttrs const &tensor) {
    return parallel_tensor_attrs_from_tensor_attrs(tensor);
  };
  auto graph_view = rewrite_value_labels(
      rewrite_node_labels(cg.raw_graph, layer_map), tensor_map);
  return ParallelComputationGraph{
      LabelledDataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>::
          create_copy_of<
              UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs,
                                                    ParallelTensorAttrs>>(
              graph_view)};
}

} // namespace FlexFlow
