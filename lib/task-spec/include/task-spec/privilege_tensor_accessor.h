#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PRIVILEGE_TENSOR_ACCESSOR_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PRIVILEGE_TENSOR_ACCESSOR_H

#include "kernels/accessor.h"
#include "task-spec/permissions.h"

namespace FlexFlow {

template <Permissions>
struct privilege_mode_to_accessor_t {};

template <>
struct privilege_mode_to_accessor_t<Permissions::RW> {
  using type = GenericTensorAccessorW;
};

template <>
struct privilege_mode_to_accessor_t<Permissions::RO> {
  using type = GenericTensorAccessorR;
};

template <>
struct privilege_mode_to_accessor_t<Permissions::WO> {
  using type = GenericTensorAccessorW;
};

template <Permissions PRIV>
using privilege_mode_to_accessor =
    typename privilege_mode_to_accessor_t<PRIV>::type;

using GenericTensorAccessor =
    std::variant<GenericTensorAccessorR, GenericTensorAccessorW>;
using VariadicGenericTensorAccessor =
    std::variant<std::vector<GenericTensorAccessorR>,
                 std::vector<GenericTensorAccessorW>>;

} // namespace FlexFlow

#endif
