#include "compiler/task_signature_tensor_key.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/containers/transform.h"
#include "utils/containers/set_union.h"

namespace FlexFlow {

std::unordered_set<TaskSignatureTensorKey>
  all_keys_for_signature_arities(
    nonnegative_int num_inputs,
    nonnegative_int num_weights,
    nonnegative_int num_outputs) {
  
  auto mk_key_set = [](nonnegative_int num, TensorRole role) {
    return transform(unordered_set_of(nonnegative_range(num)),
                     [&](nonnegative_int idx) {
                       return TaskSignatureTensorKey{
                         /*tensor_role=*/role,
                         /*idx=*/idx,
                       };
                     });
  };


  return set_union(std::vector{
    mk_key_set(num_inputs, TensorRole::INPUT),
    mk_key_set(num_weights, TensorRole::WEIGHT),
    mk_key_set(num_outputs, TensorRole::OUTPUT),
  });
}


} // namespace FlexFlow
