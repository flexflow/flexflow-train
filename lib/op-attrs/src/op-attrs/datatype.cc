#include "op-attrs/datatype.h"
#include "utils/containers/contains.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

nonnegative_int size_of_datatype(DataType data_type) {
  switch (data_type) {
    case DataType::BOOL:
      return nonnegative_int{sizeof(bool)};
    case DataType::INT32:
      return nonnegative_int{sizeof(int32_t)};
    case DataType::INT64:
      return nonnegative_int{sizeof(int64_t)};
    case DataType::HALF:
      return nonnegative_int{sizeof(float)} / 2_n;
    case DataType::FLOAT:
      return nonnegative_int{sizeof(float)};
    case DataType::DOUBLE:
      return nonnegative_int{sizeof(double)};
    default:
      throw mk_runtime_error(fmt::format("Unknown DataType {}", data_type));
  }
}

bool can_strictly_promote_datatype_from_to(DataType src, DataType dst) {
  std::unordered_set<DataType> allowed;
  switch (src) {
    case DataType::BOOL:
      allowed = {
          DataType::INT32, DataType::INT64, DataType::FLOAT, DataType::DOUBLE};
      break;
    case DataType::INT32:
      allowed = {DataType::INT64};
      break;
    case DataType::INT64:
      break;
    case DataType::HALF:
      allowed = {DataType::FLOAT, DataType::DOUBLE};
      break;
    case DataType::FLOAT:
      allowed = {DataType::DOUBLE};
      break;
    case DataType::DOUBLE:
      break;
    default:
      throw mk_runtime_error(fmt::format("Unknown DataType {}", src));
  }

  return contains(allowed, dst);
}

} // namespace FlexFlow
