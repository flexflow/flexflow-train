#ifndef _FLEXFLOW_RUNTIME_SRC_OP_TASK_SIGNATURE_H
#define _FLEXFLOW_RUNTIME_SRC_OP_TASK_SIGNATURE_H

#include "utils/visitable.h"
#include "utils/type_index.h"
#include "slot_type.h"
#include "slot_id.h"
#include "serialization.h"
#include "tasks.h"
#include "utils/type_index.h"

namespace FlexFlow {

enum class TensorRole {
  INPUT,
  WEIGHT,
  OUTPUT,
};

enum class OpTaskType { INIT, FWD, BWD };
enum class IsGrad { YES, NO };

enum class OpSlotOptions {
  OPTIONAL,
  UNTRAINABLE,
  OPTIONAL_UNTRAINABLE,
  NECESSARY
};

struct OpTensorSlotSpec {
public:
  OpTensorSlotSpec() = delete;

public:
  slot_id name;
  SlotType slot_type;
  TensorRole tensor_role;
  IsGrad is_grad;
  OpSlotOptions slot_option;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(
    OpTensorSlotSpec, name, slot_type, tensor_role, is_grad, slot_option);

struct OpTaskSignature {
  OpTaskSignature() = default;
  // explicit OpTaskSignature(OpTaskType);

  OpTaskType get_task_type() const {
    return this->type;
  }

  void add_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_optional_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_untrainable_input_slot(slot_id,
                                  SlotType slot_type = SlotType::TENSOR);
  void add_optional_untrainable_input_slot(
      slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_output_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_bwd_necessary_output_slot(slot_id,
                                     SlotType slot_type = SlotType::TENSOR);

  void add_weight_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_optional_weight_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_from_slot_spec(OpTensorSlotSpec const &spec);

  template <typename T>
  void add_arg_slot(slot_id name) {
    static_assert(is_serializable<T>::value, "Type must be serializable");
    this->task_arg_types.insert({name, init_type_index<T>()});
  }

  template <typename T>
  void add_return_value() {
    //std::type_index return_value = init_type_index<T>();
    this->return_value = init_type_index<T>();
  }

  // adds arg_slot without checking is_serializable, used for arguments that are
  // deviceSpecific
  template <typename T>
  void add_unchecked_arg_slot(slot_id name) {
    this->task_arg_types.insert({name, init_type_index<T>()});
  }

  std::unordered_set<OpTensorSlotSpec> get_tensor_slots();
  void set_arg_types(std::unordered_map<slot_id, std::type_index> const &);
  std::unordered_map<slot_id, std::type_index> get_arg_types();

  OpTaskType type;
  std::optional<std::type_index> return_value;
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_set<OpTensorSlotSpec> op_tensor_slots;
};
// FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(OpTaskSignature, 
//                                             type,  
//                                             return_value,
//                                             task_arg_types, 
//                                             op_tensor_slots);

template <typename F>
void register_task(task_id_t,
                   std::string const &name,
                   OpTaskSignature const &,
                   F const &func);

template <typename F>
void register_task(task_id_t,
                   std::string const &name,
                   OpTaskSignature const &,
                   F const &func,
                   F const &cpu_func);

template <task_id_t>
OpTaskSignature init_signature();

template <task_id_t>
OpTaskSignature fwd_signature();

template <task_id_t>
OpTaskSignature bwd_signature();

} // namespace FlexFlow

#endif