#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "kernels/accessor.h"
#include "pcg/device_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/task_argument_accessor/itask_argument_accessor.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.dtg.h"
#include <unordered_map>

namespace FlexFlow {

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
  explicit LocalTaskArgumentAccessor(
      Allocator const &allocator,
      std::unordered_map<TaskTensorParameter, DynamicTensorAccessor> const
          &tensor_slots_backing,
      ProfilingSettings const &profiling_settings,
      device_handle_t const &ff_handle,
      DeviceType kernel_device_type,
      PCGOperatorAttrs const &op_attrs,
      std::optional<LossAttrs> const &loss_attrs,
      std::optional<PerDeviceOpState> const &per_device_op_state,
      FFIterationConfig const &iteration_config,
      std::optional<OptimizerAttrs> const &optimizer_attrs,
      size_t device_idx);

  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor const &) = delete;
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor &&) = delete;

  ConcreteArgSpec const &get_concrete_arg(arg_slot_id_t) const override;

  GenericTensorAccessor get_tensor(TaskTensorParameter slot,
                                   Permissions priv) const override;

  ProfilingSettings get_profiling_settings() const override;
  device_handle_t get_ff_handle() const override;
  DeviceType get_kernel_device_type() const override;
  PCGOperatorAttrs get_op_attrs() const override;
  LossAttrs get_loss_attrs() const override;
  PerDeviceOpState get_per_device_op_state() const override;
  FFIterationConfig get_iteration_config() const override;
  OptimizerAttrs get_optimizer_attrs() const override;

  Allocator get_allocator() const override;

  device_id_t get_device_idx() const override;

private:
  Allocator allocator;
  std::unordered_map<TaskTensorParameter, DynamicTensorAccessor>
      tensor_slots_backing;

  ProfilingSettings profiling_settings;
  device_handle_t ff_handle;
  DeviceType kernel_device_type;
  PCGOperatorAttrs op_attrs;
  std::optional<LossAttrs> loss_attrs;
  std::optional<PerDeviceOpState> per_device_op_state;
  FFIterationConfig iteration_config;
  std::optional<OptimizerAttrs> optimizer_attrs;

  device_id_t device_idx;
};

CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif
