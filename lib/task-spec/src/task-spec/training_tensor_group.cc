#include "task-spec/training_tensor_group.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/repeat.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/set_union.h"

namespace FlexFlow {

TrainingTensorGroup 
  make_training_tensor_group_for_tensor_guid_t(tensor_guid_t tensor_guid,
                                               TensorAttrs const &tensor_attrs,
                                               OptimizerAttrs const &optimizer_attrs,
                                               ForwardTensorSource &forward_tensor_source,
                                               GradientTensorSource &gradient_tensor_source,
                                               OptimizerTensorSource &optimizer_tensor_source) {

  nonnegative_int num_optimizer_tensors = [&]() {
    if (tensor_attrs.create_grad == CreateGrad::YES) {
      return get_num_optimizer_tensors(optimizer_attrs);
    } else {
      return 0_n;
    }
  }();

  return TrainingTensorGroup{
    /*forward_tensor=*/forward_tensor_source.new_forward_tensor(),
    /*gradient_tensor=*/gradient_tensor_source.new_gradient_tensor(),
    /*optimizer_tensors=*/repeat(
      num_optimizer_tensors,
      [&]() { return optimizer_tensor_source.new_optimizer_tensor(); }),
  };
}

std::unordered_set<training_tensor_guid_t> get_all_training_tensors_in_tensor_group(TrainingTensorGroup const &group) {
  return set_union(
    std::unordered_set{
      training_tensor_guid_t{group.forward_tensor},
      training_tensor_guid_t{group.gradient_tensor},
    },
    transform(unordered_set_of(group.optimizer_tensors), 
              [](optimizer_tensor_guid_t optimizer_tensor) { return training_tensor_guid_t{optimizer_tensor}; })
  );
}

} // namespace FlexFlow
