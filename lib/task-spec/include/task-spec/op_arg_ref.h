#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_ARG_REF_H

#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "task-spec/arg_ref.h"
#include "task-spec/device_specific.h"
#include "task-spec/op_arg_ref_type.dtg.h"
#include "task-spec/per_device_op_state.h"

namespace FlexFlow {

template <typename T>
using OpArgRef = ArgRef<OpArgRefType, T>;

using OpArgRefSpec = ArgRefSpec<OpArgRefType>;

template <typename T>
OpArgRef<DeviceSpecificDeviceStates> per_device_op_state() {
  OpArgRefType op_arg_ref_type = OpArgRefType{PerDeviceOpStateRefType{}};
  static_assert(PerDeviceOpState::IsPartOfPerDeviceOpState_v<T>);
  ArgRef<OpArgRefType, DeviceSpecificDeviceStates> arg_ref = {op_arg_ref_type};
  return arg_ref;
}

OpArgRef<ParallelTensorShape> input_parallel_tensor_shape(nonnegative_int idx);
OpArgRef<ParallelTensorShape> weight_parallel_tensor_shape(nonnegative_int idx);
OpArgRef<ParallelTensorShape> output_parallel_tensor_shape(nonnegative_int idx);

} // namespace FlexFlow

#endif
