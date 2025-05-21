#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H

#include "op-attrs/datatype.dtg.h"
#include "utils/fmt.h"
#include "utils/fp16.h"
#include "utils/positive_int/positive_int.h"
#include <variant>

namespace FlexFlow {

template <DataType>
struct data_type_enum_to_class;

template <>
struct data_type_enum_to_class<DataType::FLOAT> : type_identity<float> {};

template <>
struct data_type_enum_to_class<DataType::DOUBLE> : type_identity<double> {};

template <>
struct data_type_enum_to_class<DataType::INT32> : type_identity<int32_t> {};

template <>
struct data_type_enum_to_class<DataType::INT64> : type_identity<int64_t> {};

template <>
struct data_type_enum_to_class<DataType::HALF> : type_identity<half> {};

template <>
struct data_type_enum_to_class<DataType::BOOL> : type_identity<bool> {};

template <typename T>
struct type_to_data_type_enum;

template <>
struct type_to_data_type_enum<double>
    : std::integral_constant<DataType, DataType::DOUBLE> {};

template <>
struct type_to_data_type_enum<float>
    : std::integral_constant<DataType, DataType::FLOAT> {};

template <>
struct type_to_data_type_enum<half>
    : std::integral_constant<DataType, DataType::HALF> {};

template <>
struct type_to_data_type_enum<int32_t>
    : std::integral_constant<DataType, DataType::INT32> {};

template <>
struct type_to_data_type_enum<int64_t>
    : std::integral_constant<DataType, DataType::INT64> {};

template <>
struct type_to_data_type_enum<bool>
    : std::integral_constant<DataType, DataType::BOOL> {};

template <typename T>
inline constexpr DataType type_to_data_type_enum_v =
    type_to_data_type_enum<T>::value;

template <DataType DT, typename T>
typename data_type_enum_to_class<DT>::type cast_to(T t) {
  return (typename data_type_enum_to_class<DT>::type)t;
}

template <DataType DT>
using real_type_t = typename data_type_enum_to_class<DT>::type;

positive_int size_of_datatype(DataType);

/**
 * @brief Maximally semantics-preserving casts, not including identity
 * casts (e.g., `float -> float` returns `false`)
 */
bool can_strictly_promote_datatype_from_to(DataType from, DataType to);

/**
 * @brief Equivalent to
 * [`torch.can_cast`](https://pytorch.org/docs/stable/generated/torch.can_cast.html),
 * except that identity casts (e.g., `float -> float`) return `false`
 */
bool can_torch_strictly_promote_datatype_from_to(DataType from, DataType to);

} // namespace FlexFlow

#endif
