#include "op-attrs/datatype_value.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

DataTypeValue make_half_data_type_value(half value) {
  return DataTypeValue{value};
}

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
      [](half) { return DataType::HALF; },
      [](float) { return DataType::FLOAT; },
      [](double) { return DataType::DOUBLE; },
      [](int32_t) { return DataType::INT32; },
      [](int64_t) { return DataType::INT64; },
      [](bool) { return DataType::BOOL; },
  });
}

DataTypeValue make_zero_data_type_value_of_type(DataType data_type) {
  std::optional<DataTypeValue> result = std::nullopt;

  switch (data_type) {
    case DataType::HALF:
      result = make_half_data_type_value(0.0);
      break;
    case DataType::FLOAT:
      result = make_float_data_type_value(0.0);
      break;
    case DataType::DOUBLE:
      result = make_double_data_type_value(0.0);
      break;
    case DataType::INT32:
      result = make_int32_data_type_value(0);
      break;
    case DataType::INT64:
      result = make_int64_data_type_value(0);
      break;
    case DataType::BOOL:
      result = make_bool_data_type_value(false);
      break;
    default:
      PANIC("Unhandled DataType value", data_type);
  };

  ASSERT(result.has_value());
  ASSERT(get_data_type_of_data_type_value(result.value()) == data_type);

  return result.value();
}

} // namespace FlexFlow
