#include "task-spec/ops/op_arg_ref.h"

namespace FlexFlow {

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(nonnegative_int idx) {
  OpArgRefType arg_ref_type = OpArgRefType{ParallelTensorShapeRefType{
      /*tensor_role=*/TensorRole::INPUT,
      /*idx=*/idx,
  }};
  ArgRef<OpArgRefType, ParallelTensorShape> arg_ref = {arg_ref_type};
  return arg_ref;
}

OpArgRef<ParallelTensorShape>
    weight_parallel_tensor_shape(nonnegative_int idx) {
  OpArgRefType arg_ref_type = OpArgRefType{ParallelTensorShapeRefType{
      /*tensor_role=*/TensorRole::WEIGHT,
      /*idx=*/idx,
  }};
  ArgRef<OpArgRefType, ParallelTensorShape> arg_ref = {arg_ref_type};
  return arg_ref;
}

OpArgRef<ParallelTensorShape>
    output_parallel_tensor_shape(nonnegative_int idx) {
  OpArgRefType arg_ref_type = OpArgRefType{ParallelTensorShapeRefType{
      /*tensor_role=*/TensorRole::OUTPUT,
      /*idx=*/idx,
  }};
  ArgRef<OpArgRefType, ParallelTensorShape> arg_ref = {arg_ref_type};
  return arg_ref;
}

} // namespace FlexFlow
