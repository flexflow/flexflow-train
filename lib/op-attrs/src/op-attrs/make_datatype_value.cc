#include "op-attrs/make_datatype_value.h"

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

} // namespace FlexFlow
