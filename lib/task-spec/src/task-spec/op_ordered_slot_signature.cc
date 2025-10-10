#include "task-spec/op_ordered_slot_signature.h"
#include "utils/containers/filtermap_values.h"
#include "utils/containers/repeat.h"

namespace FlexFlow {

OpOrderedSlotSignature get_op_ordered_slot_signature_for_binding(OpTaskBinding const &binding, 
                                                                 nonnegative_int num_inputs,
                                                                 nonnegative_int num_weights,
                                                                 nonnegative_int num_outputs) {
  auto get_ordered_slots_for_role = [&](TensorRole tensor_role) 
    -> std::unordered_map<fwb_tensor_slot_id_t, nonnegative_int>
  {
    return filtermap_values(binding.get_tensor_bindings(),
                     [&](OpTensorSpec const &op_tensor_spec) -> std::optional<nonnegative_int> {
                       if (op_tensor_spec.role == tensor_role) {
                         return op_tensor_spec.idx; 
                       } else {
                         return std::nullopt;
                       }
                     });
  };

  auto to_set_vector = [](nonnegative_int num, std::unordered_map<fwb_tensor_slot_id_t, nonnegative_int> const &m) 
    -> std::vector<std::unordered_set<fwb_tensor_slot_id_t>>
  {
    std::vector<std::unordered_set<fwb_tensor_slot_id_t>> set_vector 
      = repeat(num, []() { return std::unordered_set<fwb_tensor_slot_id_t>{}; });
    for (auto const &[slot, idx] : m) {
      set_vector.at(idx.unwrap_nonnegative()).insert(slot);
    }
    return set_vector;
  };

  return OpOrderedSlotSignature{
    /*input_slots=*/to_set_vector(num_inputs, get_ordered_slots_for_role(TensorRole::INPUT)),
    /*weight_slots=*/to_set_vector(num_weights, get_ordered_slots_for_role(TensorRole::WEIGHT)),
    /*output_slots=*/to_set_vector(num_outputs, get_ordered_slots_for_role(TensorRole::OUTPUT)),
  };
}



} // namespace FlexFlow
