#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DATATYPE_H

#include "op-attrs/datatype.dtg.h"
#include "utils/fmt.h"
#include "utils/fp16.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <variant>

namespace FlexFlow {

template <DataType>
struct data_type_enum_to_class;

template <>
struct data_type_enum_to_class<DataType::FLOAT> {
  using type = float;
};

template <>
struct data_type_enum_to_class<DataType::DOUBLE> {
  using type = double;
};

template <>
struct data_type_enum_to_class<DataType::INT32> {
  using type = int32_t;
};

template <>
struct data_type_enum_to_class<DataType::INT64> {
  using type = int64_t;
};

template <>
struct data_type_enum_to_class<DataType::HALF> {
  using type = half;
};

template <>
struct data_type_enum_to_class<DataType::BOOL> {
  using type = bool;
};

template <DataType DT, typename T>
typename data_type_enum_to_class<DT>::type cast_to(T t) {
  return (typename data_type_enum_to_class<DT>::type)t;
}

template <DataType DT>
using real_type_t = typename data_type_enum_to_class<DT>::type;

nonnegative_int size_of_datatype(DataType);

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
