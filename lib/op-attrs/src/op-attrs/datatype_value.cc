#include "op-attrs/datatype_value.h"
#include "utils/overload.h"

namespace FlexFlow {

DataTypeValue make_float_data_type_value(float value) {
  return DataTypeValue{value};
}

DataTypeValue make_double_data_type_value(double value) {
  return DataTypeValue{value};
}

DataTypeValue make_int32_data_type_value(int32_t value) {
  return DataTypeValue{value};
}

DataTypeValue make_int64_data_type_value(int64_t value) {
  return DataTypeValue{value};
}

DataTypeValue make_bool_data_type_value(bool value) {
  return DataTypeValue{value};
}

DataType get_data_type_of_data_type_value(DataTypeValue value) {
  return value.visit<DataType>(overload{
      [](float) { return DataType::FLOAT; },
      [](double) { return DataType::DOUBLE; },
      [](int32_t) { return DataType::INT32; },
      [](int64_t) { return DataType::INT64; },
      [](bool) { return DataType::BOOL; },
  });
}

} // namespace FlexFlow
