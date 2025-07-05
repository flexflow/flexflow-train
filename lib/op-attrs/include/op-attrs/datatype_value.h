#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DATATYPE_VALUE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DATATYPE_VALUE_H

#include "op-attrs/datatype.dtg.h"
#include "op-attrs/datatype_value.dtg.h"
#include "utils/half.h"

namespace FlexFlow {

DataTypeValue make_half_data_type_value(half value);
DataTypeValue make_float_data_type_value(float value);
DataTypeValue make_double_data_type_value(double value);
DataTypeValue make_int32_data_type_value(int32_t value);
DataTypeValue make_int64_data_type_value(int64_t value);
DataTypeValue make_bool_data_type_value(bool value);

DataTypeValue make_zero_data_type_value_of_type(DataType);

DataType get_data_type_of_data_type_value(DataTypeValue);

} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_MAKE_DATATYPE_VALUE_H
