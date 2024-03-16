#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_TENSOR_H

#include "data_type.h"
#include "initializer.h"
#include "op-attrs/tensor_shape.h"
#include "param_sync.h"
#include "pcg/tensor.h"
#include "utils/visitable.h"
#include <vector>

namespace FlexFlow {

struct V1TensorShape {
  std::vector<size_t> dims;
  req<V1DataType> data_type;
};
FF_VISITABLE_STRUCT(V1TensorShape, dims, data_type);
CHECK_IS_JSONABLE(V1TensorShape);
V1TensorShape to_v1(TensorShape const &);

struct V1Tensor {
  V1TensorShape shape;
  std::optional<V1Initializer> initializer;
  bool create_gradients;
  std::optional<V1ParamSync> sync_type;
  req<std::optional<std::string>> name;
};
FF_VISITABLE_STRUCT(
    V1Tensor, shape, initializer, create_gradients, sync_type, name);
CHECK_IS_JSONABLE(V1Tensor);
V1Tensor to_v1(Tensor const &);

} // namespace FlexFlow

#endif