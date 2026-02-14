#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_TASK_ARG_SERIALIZER_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_TASK_ARG_SERIALIZER_H

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace FlexFlow {

template <typename T>
std::string serialize_task_args(T const &args) {
  nlohmann::json j = args;
  return j.dump();
}

template <typename T>
T deserialize_task_args(void const *args, size_t arglen) {
  nlohmann::json j = nlohmann::json::parse(
      std::string_view{reinterpret_cast<char const *>(args), arglen});
  return j.get<T>();
}

} // namespace FlexFlow

#endif
