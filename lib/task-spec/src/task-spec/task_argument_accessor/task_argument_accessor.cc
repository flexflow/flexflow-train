#include "task-spec/task_argument_accessor/task_argument_accessor.h"

namespace FlexFlow {

ProfilingSettings TaskArgumentAccessor::get_profiling_settings() const { 
  NOT_IMPLEMENTED();
}

device_handle_t TaskArgumentAccessor::get_ff_handle() const { 
  NOT_IMPLEMENTED();
}
DeviceType TaskArgumentAccessor::get_kernel_device_type() const {
  NOT_IMPLEMENTED();
}

PCGOperatorAttrs TaskArgumentAccessor::get_op_attrs() const {
  NOT_IMPLEMENTED();
}

PerDeviceOpState TaskArgumentAccessor::get_per_device_op_state() const {
  NOT_IMPLEMENTED();
}

FFIterationConfig TaskArgumentAccessor::get_iteration_config() const {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
