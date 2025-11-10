#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "pcg/operator_space_to_machine_space_mapping.dtg.h"
#include "utils/bidict/bidict.h"
#include <cstddef>
#include <optional>
#include <unordered_set>

namespace FlexFlow {

nonnegative_int mv_get_expected_task_space_num_dims(MachineView const &mv);

DeviceType get_device_type(MachineView const &mv);

std::vector<stride_t> get_strides(MachineView const &mv);

std::vector<MachineSpecificationDimension>
    get_dimensions(MachineView const &mv);

MachineView machine_view_from_strides_and_machine_spec_dimensions(
    MachineSpaceCoordinate const &start,
    std::vector<stride_t> const &strides,
    std::vector<MachineSpecificationDimension> const &dims);

MachineSpaceCoordinate
    get_machine_space_coordinate(OperatorTaskSpace const &operator_task_space,
                                 MachineView const &machine_view,
                                 TaskSpaceCoordinate const &task_space_coordinate);

TaskSpaceCoordinate
    mv_task_space_coord_for_machine_space_coord(
      MachineView const &,
      OperatorTaskSpace const &,
      MachineSpaceCoordinate const &);

OperatorSpaceToMachineSpaceMapping
  get_coordinate_mapping_for_machine_view(OperatorTaskSpace const &operator_task_space,
                                          MachineView const &machine_view);

std::unordered_set<MachineSpaceCoordinate>
    get_machine_space_coordinates(OperatorTaskSpace const &task,
                                  MachineView const &mv);

std::unordered_set<device_id_t> get_device_ids(OperatorTaskSpace const &task,
                                               MachineView const &mv,
                                               MachineComputeSpecification const &ms);

MachineView make_1d_machine_view(MachineSpaceCoordinate const &start,
                                 MachineSpecificationDimension const &dim,
                                 stride_t stride);

MachineView make_single_device_machine_view(MachineSpaceCoordinate const &);

OperatorAtomicTaskShardBinding
  operator_atomic_task_shard_binding_from_machine_view(ComputationGraphOpAttrs const &,
                                                       std::vector<ParallelTensorDimDegrees> const &,
                                                       MachineView const &,
                                                       MachineSpaceCoordinate const &);

MappedOperatorTaskGroup
  mapped_operator_task_group_from_machine_view(
    ComputationGraphOpAttrs const &,
    std::vector<ParallelTensorDimDegrees> const &,
    MachineView const &);

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> 
  get_tensor_shard_to_device_coord_mapping(ComputationGraphOpAttrs const &,
                                           MachineView const &);

} // namespace FlexFlow

#endif
