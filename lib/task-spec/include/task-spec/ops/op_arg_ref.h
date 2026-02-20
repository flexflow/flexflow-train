#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_ARG_REF_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_ARG_REF_H

#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "task-spec/arg_ref.h"
#include "task-spec/device_specific.h"
#include "task-spec/op_arg_ref_type.dtg.h"
#include "task-spec/per_device_op_state.h"

namespace FlexFlow {

template <typename T>
using OpArgRef = ArgRef<OpArgRefType, T>;

template <typename T>
OpArgRef<T> per_device_op_state() {
  OpArgRefType op_arg_ref_type = OpArgRefType{PerDeviceOpStateRefType{}};
  static_assert(PerDeviceOpState::IsPartOfPerDeviceOpState_v<T>);
  ArgRef<OpArgRefType, T> arg_ref = {op_arg_ref_type};
  return arg_ref;
}

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(nonnegative_int idx);
OpArgRef<ParallelTensorShape> weight_parallel_tensor_shape(nonnegative_int idx);
OpArgRef<ParallelTensorShape> output_parallel_tensor_shape(nonnegative_int idx);

} // namespace FlexFlow

#endif
