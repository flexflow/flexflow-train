#include "task-spec/training_symbolic_computation_graph_from_cg_conversion.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "task-spec/symbolic_training_tensor_group.h"
#include "task-spec/symbolic_training_tensor_group_with_attrs.h"
#include "task-spec/training_symbolic_computation_graph.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_value_labels.h"

namespace FlexFlow {

TrainingSymbolicComputationGraphFromCgConversion generate_training_computation_graph_from_cg(
    ComputationGraph const &computation_graph,
    OptimizerAttrs const &optimizer_attrs,
    tensor_guid_t const &logit_tensor,
    SymbolicForwardTensorSource &forward_tensor_source,
    SymbolicGradientTensorSource &gradient_tensor_source,
    SymbolicOptimizerTensorSource &optimizer_tensor_source,
    SymbolicLossTensorSource &loss_tensor_source) {

  symbolic_loss_tensor_guid_t label_tensor = loss_tensor_source.new_symbolic_loss_tensor();

  LabelledDataflowGraphView<ParallelLayerAttrs, TensorAttrs>
    with_nodes_relabelled = 
        rewrite_node_labels(computation_graph.raw_graph,
                            [](Node const &, LayerAttrs const &layer_attrs) -> ParallelLayerAttrs {
                              return parallel_layer_attrs_from_layer_attrs(layer_attrs);
                            });

  LabelledDataflowGraphView<ParallelLayerAttrs, TensorShape>
    symbolic_computation_graph = 
      rewrite_value_labels(
        with_nodes_relabelled,
        [](OpenDataflowValue const &, TensorAttrs const &tensor_attrs) -> TensorShape {
          return tensor_attrs.shape;
        });

  bidict<tensor_guid_t, symbolic_tensor_guid_t>
    tensor_mapping = generate_bidict(get_all_tensors(computation_graph),
                                     [](tensor_guid_t t) {
                                       return symbolic_tensor_guid_t{t.raw_graph_output};
                                     });

  bidict<layer_guid_t, symbolic_layer_guid_t>
    layer_mapping = generate_bidict(get_layers(computation_graph),
                                    [](layer_guid_t l) {
                                      return symbolic_layer_guid_t{l.raw_node};
                                    });

  TrainingSymbolicComputationGraph training_symbolic_cg = TrainingSymbolicComputationGraph{
      /*symbolic_computation_graph=*/symbolic_computation_graph,
      /*symbolic_training_tensor_group_for_tensor=*/
      transform(
          get_all_tensor_attrs(computation_graph),
          [&](tensor_guid_t tensor_guid, TensorAttrs const &tensor_attrs) 
            -> std::pair<symbolic_tensor_guid_t, SymbolicTrainingTensorGroup>
          {
            return std::pair{
                tensor_mapping.at_l(tensor_guid),
                make_symbolic_training_tensor_group(
                    /*create_grad=*/tensor_attrs.create_grad,
                    /*optimizer_attrs=*/optimizer_attrs,
                    /*forward_tensor_source=*/forward_tensor_source,
                    /*gradient_tensor_source=*/gradient_tensor_source,
                    /*optimizer_tensor_source=*/optimizer_tensor_source),
            };
          }),
      /*logit_tensor=*/tensor_mapping.at_l(logit_tensor),
      /*label_tensor=*/label_tensor,
  };

  return TrainingSymbolicComputationGraphFromCgConversion{
    /*training_symbolic_computation_graph=*/training_symbolic_cg,
    /*tensor_mapping=*/tensor_mapping,
    /*layer_mapping=*/layer_mapping,
  };
}


SymbolicTrainingTensorGroup
    get_training_tensor_group_for_tensor_guid(TrainingSymbolicComputationGraphFromCgConversion const &conversion,
                                              tensor_guid_t t) {
  symbolic_tensor_guid_t symbolic_tensor_guid = conversion.tensor_mapping.at_l(t);
  return conversion.training_symbolic_computation_graph.symbolic_training_tensor_group_for_tensor.at(symbolic_tensor_guid);
}

SymbolicTrainingTensorGroupWithAttrs
    get_training_tensor_group_with_attrs_for_tensor_guid(
        TrainingSymbolicComputationGraphFromCgConversion const &conversion, 
        tensor_guid_t t) {

  symbolic_tensor_guid_t symbolic_tensor_guid = conversion.tensor_mapping.at_l(t);

  return make_symbolic_training_tensor_group_with_attrs_from_group_and_attrs(
      /*group=*/get_training_tensor_group_for_tensor_guid(conversion,
                                                          t),
      /*attrs=*/get_symbolic_tensor_shape(conversion.training_symbolic_computation_graph, symbolic_tensor_guid));
}

symbolic_layer_guid_t
    get_symbolic_layer_guid_for_layer_guid(TrainingSymbolicComputationGraphFromCgConversion const &conversion,
                                           layer_guid_t l) {
  return conversion.layer_mapping.at_l(l);
}

symbolic_tensor_guid_t
    get_symbolic_tensor_guid_for_tensor_guid(TrainingSymbolicComputationGraphFromCgConversion const &conversion,
                                            tensor_guid_t t) {
  return conversion.tensor_mapping.at_l(t);
}

layer_guid_t
    get_layer_guid_for_symbolic_layer_guid(TrainingSymbolicComputationGraphFromCgConversion const &conversion,
                                           symbolic_layer_guid_t l) {
  return conversion.layer_mapping.at_r(l);
}

} // namespace FlexFlow
