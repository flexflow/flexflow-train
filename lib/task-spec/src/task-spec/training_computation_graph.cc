#include "task-spec/training_computation_graph.h"
#include "task-spec/loss_tensor_source.h"
#include "task-spec/training_computation_graph_fragment.h"
#include "task-spec/training_tensor_group.h"
#include "task-spec/training_tensor_group_with_attrs.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"

namespace FlexFlow {

TrainingComputationGraph generate_training_computation_graph(
    ComputationGraph const &computation_graph,
    OptimizerAttrs const &optimizer_attrs,
    tensor_guid_t const &logit_tensor,
    ForwardTensorSource &forward_tensor_source,
    GradientTensorSource &gradient_tensor_source,
    OptimizerTensorSource &optimizer_tensor_source,
    LossTensorSource &loss_tensor_source) {

  loss_tensor_guid_t label_tensor = loss_tensor_source.new_loss_tensor();
    
  return TrainingComputationGraph{
      /*computation_graph=*/computation_graph,
      /*training_tensor_group_for_tensor=*/transform(
          get_all_tensor_attrs(computation_graph),
          [&](tensor_guid_t tensor_guid, TensorAttrs const &tensor_attrs) {
            return std::pair{
                tensor_guid,
                make_training_tensor_group_for_tensor_guid_t(
                    /*tensor_guid=*/tensor_guid,
                    /*tensor_attrs=*/tensor_attrs,
                    /*optimizer_attrs=*/optimizer_attrs,
                    /*forward_tensor_source=*/forward_tensor_source,
                    /*gradient_tensor_source=*/gradient_tensor_source,
                    /*optimizer_tensor_source=*/optimizer_tensor_source),
            };
          }),
      /*logit_tensor=*/logit_tensor,
      /*label_tensor=*/label_tensor,
  };
}

TrainingTensorGroup get_training_tensor_group_for_tensor_guid(
    TrainingComputationGraph const &training_cg, tensor_guid_t tensor_guid) {

  return training_cg.training_tensor_group_for_tensor.at(tensor_guid);
}

TrainingTensorGroupWithAttrs
    get_training_tensor_group_with_attrs_for_tensor_guid(
        TrainingComputationGraph const &training_cg,
        tensor_guid_t tensor_guid) {
  return make_training_tensor_group_with_attrs_from_group_and_attrs(
      /*group=*/get_training_tensor_group_for_tensor_guid(training_cg,
                                                          tensor_guid),
      /*attrs=*/get_tensor_attrs(training_cg.computation_graph, tensor_guid));
}

forward_tensor_guid_t get_forward_tensor_guid_for_tensor_guid(
    TrainingComputationGraph const &training_cg, tensor_guid_t t) {
  return training_cg.training_tensor_group_for_tensor.at(t).forward_tensor;
}

gradient_tensor_guid_t get_gradient_tensor_guid_for_tensor_guid(
    TrainingComputationGraph const &training_cg, tensor_guid_t t) {
  return training_cg.training_tensor_group_for_tensor.at(t).gradient_tensor;
}

std::vector<optimizer_tensor_guid_t> get_optimizer_tensor_guids_for_tensor_guid(
    TrainingComputationGraph const &training_cg, tensor_guid_t t) {
  return training_cg.training_tensor_group_for_tensor.at(t).optimizer_tensors;
}

tensor_guid_t get_tensor_guid_for_forward_tensor_guid(
    TrainingComputationGraph const &training_cg, forward_tensor_guid_t t) {
  return get_only(keys(filter_values(
      training_cg.training_tensor_group_for_tensor,
      [&](TrainingTensorGroup const &g) { return g.forward_tensor == t; })));
}

tensor_guid_t get_tensor_guid_for_gradient_tensor_guid(
    TrainingComputationGraph const &training_cg, gradient_tensor_guid_t t) {
  return get_only(keys(filter_values(
      training_cg.training_tensor_group_for_tensor,
      [&](TrainingTensorGroup const &g) { return g.gradient_tensor == t; })));
}

tensor_guid_t get_tensor_guid_for_optimizer_tensor_guid(
    TrainingComputationGraph const &training_cg, optimizer_tensor_guid_t t) {
  return get_only(
      keys(filter_values(training_cg.training_tensor_group_for_tensor,
                         [&](TrainingTensorGroup const &g) {
                           return contains(g.optimizer_tensors, t);
                         })));
}

tensor_guid_t get_tensor_guid_for_training_tensor_guid(
    TrainingComputationGraph const &training_cg, training_tensor_guid_t t) {
  return t.visit<tensor_guid_t>(overload{
      [&](forward_tensor_guid_t forward_tensor) {
        return get_tensor_guid_for_forward_tensor_guid(training_cg,
                                                       forward_tensor);
      },
      [&](gradient_tensor_guid_t gradient_tensor) {
        return get_tensor_guid_for_gradient_tensor_guid(training_cg,
                                                        gradient_tensor);
      },
      [&](optimizer_tensor_guid_t optimizer_tensor) {
        return get_tensor_guid_for_optimizer_tensor_guid(training_cg,
                                                         optimizer_tensor);
      },
      [&](loss_tensor_guid_t loss_tensor) -> tensor_guid_t {
        PANIC("no tensor_guid_t can exist for a loss_tensor_guid_t");
      },
  });
}

std::unordered_set<training_tensor_guid_t>
    get_all_training_tensors_in_training_computation_graph(
        TrainingComputationGraph const &training_cg) {
  std::unordered_set<training_tensor_guid_t> result = flatmap(
      unordered_set_of(keys(training_cg.training_tensor_group_for_tensor)),
      [&](tensor_guid_t t) {
        return get_all_training_tensors_in_tensor_group(
            training_cg.training_tensor_group_for_tensor.at(t));
      });
  
  result.insert(training_tensor_guid_t{training_cg.label_tensor});
  return result;
}

TrainingLayerPlusContext
    get_training_layer_plus_context(TrainingComputationGraph const &training_cg,
                                    layer_guid_t layer_guid) {
  auto get_tensor_group_with_attrs =
      [&](tensor_guid_t t) -> TrainingTensorGroupWithAttrs {
    return get_training_tensor_group_with_attrs_for_tensor_guid(training_cg, t);
  };

  return TrainingLayerPlusContext{
      /*layer_guid=*/layer_guid,
      /*layer_attrs=*/
      get_layer_attrs(training_cg.computation_graph, layer_guid),
      /*input_tensor_groups=*/
      transform(get_incoming_inputs(training_cg.computation_graph, layer_guid),
                get_tensor_group_with_attrs),
      /*weight_tensor_groups=*/
      transform(get_incoming_weights(training_cg.computation_graph, layer_guid),
                get_tensor_group_with_attrs),
      /*output_tensor_groups=*/
      transform(get_outgoing_tensors(training_cg.computation_graph, layer_guid),
                get_tensor_group_with_attrs),
  };
}

std::unordered_map<training_tensor_guid_t, TensorShape>
    get_all_training_tensor_shapes(
        TrainingComputationGraph const &training_cg) {
  std::unordered_map<training_tensor_guid_t, TensorShape> result = generate_map(
      get_all_training_tensors_in_training_computation_graph(training_cg),
      [&](training_tensor_guid_t t) {
        if (t.is_loss_tensor()) {
          ASSERT(t == training_tensor_guid_t{training_cg.label_tensor});
          return get_tensor_attrs(training_cg.computation_graph, 
                                  training_cg.logit_tensor).shape;
        }

        return get_tensor_attrs(
                   training_cg.computation_graph,
                   get_tensor_guid_for_training_tensor_guid(training_cg, t))
            .shape;
      });
}

} // namespace FlexFlow
