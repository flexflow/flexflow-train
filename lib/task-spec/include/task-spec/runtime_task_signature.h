#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_SIGNATURE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_SIGNATURE_H

#include "task-spec/runtime_task_signature.dtg.h"
#include "utils/type_index.h"

namespace FlexFlow {

RuntimeTaskSignature make_empty_runtime_task_signature();

void add_slot(RuntimeTaskSignature &,
              int name,
              TrainingTensorType,
              SlotType slot_type = SlotType::TENSOR);
void add_slot(RuntimeTaskSignature &,
              slot_id_t name,
              TrainingTensorType,
              SlotType slot_type = SlotType::TENSOR);

template <typename T>
void add_arg_slot(RuntimeTaskSignature &task_signature, int name) {
  add_arg_slot<T>(task_signature, slot_id_t{name});
}

template <typename T>
void add_arg_slot(RuntimeTaskSignature &task_signature, slot_id_t name) {
  // static_assert(is_serializable<T>::value, "Type must be serializable");
  task_signature.task_arg_types.insert({name, get_type_index_for_type<T>()});
}

template <typename T>
void add_return_value(RuntimeTaskSignature &task_signature) {
  task_signature.return_value = get_type_index_for_type<T>();
}

/**
 * @brief Adds an argument slot without checking if it is serializable.
 *
 * This function is used for arguments that are device-specific.
 */

template <typename T>
void add_unchecked_arg_slot(RuntimeTaskSignature &task_signature, int name) {
  add_unchecked_arg_slot<T>(task_signature, slot_id_t{name});
}

/**
 * @brief Adds an argument slot without checking if it is serializable.
 *
 * This function is used for arguments that are device-specific.
 */

template <typename T>
void add_unchecked_arg_slot(RuntimeTaskSignature &task_signature, slot_id_t name) {
  task_signature.task_arg_types.insert({name, get_type_index_for_type<T>()});
}

} // namespace FlexFlow

#endif
