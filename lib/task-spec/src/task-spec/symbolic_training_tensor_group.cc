#include "task-spec/symbolic_training_tensor_group.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/repeat.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"

namespace FlexFlow {

SymbolicTrainingTensorGroup make_symbolic_training_tensor_group_for_tensor_guid_t(
    CreateGrad create_grad,
    OptimizerAttrs const &optimizer_attrs,
    SymbolicForwardTensorSource &forward_tensor_source,
    SymbolicGradientTensorSource &gradient_tensor_source,
    SymbolicOptimizerTensorSource &optimizer_tensor_source) {

  nonnegative_int num_optimizer_tensors = [&]() {
    if (create_grad == CreateGrad::YES) {
      return get_num_optimizer_tensors(optimizer_attrs);
    } else {
      return 0_n;
    }
  }();

  return SymbolicTrainingTensorGroup{
      /*forward_tensor=*/forward_tensor_source.new_symbolic_forward_tensor(),
      /*gradient_tensor=*/gradient_tensor_source.new_symbolic_gradient_tensor(),
      /*optimizer_tensors=*/
      repeat(num_optimizer_tensors,
             [&]() { return optimizer_tensor_source.new_symbolic_optimizer_tensor(); }),
  };
}

std::unordered_set<symbolic_training_tensor_guid_t>
    get_all_training_tensors_in_tensor_group(SymbolicTrainingTensorGroup const &group) {
  return set_union(
      std::unordered_set{
          symbolic_training_tensor_guid_t{group.forward_tensor},
          symbolic_training_tensor_guid_t{group.gradient_tensor},
      },
      transform(unordered_set_of(group.optimizer_tensors),
                [](symbolic_optimizer_tensor_guid_t optimizer_tensor) {
                  return symbolic_training_tensor_guid_t{optimizer_tensor};
                }));
}

} // namespace FlexFlow
