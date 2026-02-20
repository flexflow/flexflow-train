#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SERIALIZATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SERIALIZATION_H

#include "kernels/device.h"
#include "kernels/nccl.h"
#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/required.h"
#include "utils/type_traits.h"
#include "utils/variant.h"

namespace FlexFlow {

template <typename T>
struct needs_serialization {};

template <typename... Args>
struct visit_trivially_serializable;

template <typename T, typename Enable = void>
struct is_trivially_serializable : std::false_type {};

template <typename T, typename... Args>
struct visit_trivially_serializable<T, Args...> {
  static constexpr bool value = is_trivially_serializable<T>::value &&
                                visit_trivially_serializable<Args...>::value;
};

template <typename... Args>
struct visit_trivially_serializable<std::tuple<Args...>> {
  static constexpr bool value = visit_trivially_serializable<Args...>::value;
};

template <>
struct visit_trivially_serializable<> : std::true_type {};

template <typename T>
struct is_trivially_serializable<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type>
    : std::true_type {};

template <>
struct is_trivially_serializable<half> : std::true_type {};
template <>
struct is_trivially_serializable<ncclUniqueId> : std::true_type {};

template <typename T>
struct is_trivially_serializable<
    T,
    typename std::enable_if<std::is_enum<T>::value>::type> : std::true_type {};

template <typename T>
struct is_trivially_serializable<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type>
    : std::true_type {};

template <typename T, std::size_t MAXSIZE>
struct is_trivially_serializable<stack_vector<T, MAXSIZE>>
    : is_trivially_serializable<T> {};

template <typename T>
struct is_trivially_serializable<FFOrdered<T>> : is_trivially_serializable<T> {
};

template <typename... Ts>
struct is_trivially_serializable<std::variant<Ts...>>
    : elements_satisfy<is_trivially_serializable, std::variant<Ts...>> {};

template <typename T>
struct is_trivially_serializable<std::optional<T>>
    : is_trivially_serializable<T> {};

template <typename T>
struct std_array_size_helper;

template <typename T, std::size_t N>
struct std_array_size_helper<std::array<T, N>> {
  static const std::size_t value = N;
};

template <typename T>
using std_array_size = std_array_size_helper<T>;

template <typename T>
struct is_trivially_serializable<
    T,
    std::enable_if<std::is_same<
        T,
        std::array<typename T::value_type, std_array_size<T>::value>>::value>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_serializable : std::false_type {};

template <typename T>
struct is_serializable<
    T,
    typename std::enable_if<is_trivially_serializable<T>::value>::type>
    : std::true_type {};

static_assert(is_trivially_serializable<float>::value, "");
static_assert(is_trivially_serializable<double>::value, "");
static_assert(is_trivially_serializable<int32_t>::value, "");
static_assert(is_trivially_serializable<int64_t>::value, "");
static_assert(is_trivially_serializable<half>::value, "");
static_assert(is_trivially_serializable<bool>::value, "");
static_assert(is_trivially_serializable<std::variant<float, double>>::value,
              "");

} // namespace FlexFlow

#endif
