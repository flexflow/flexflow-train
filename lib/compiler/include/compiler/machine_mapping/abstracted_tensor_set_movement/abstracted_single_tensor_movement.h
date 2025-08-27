#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_SINGLE_TENSOR_MOVEMENT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_SINGLE_TENSOR_MOVEMENT_H

#include "compiler/cost_estimator/single_tensor_movement.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"

namespace FlexFlow {

SingleTensorMovement 
  concretize_abstracted_single_tensor_movement(
    AbstractedSingleTensorMovement const &abstracted,
    MachineComputeSpecification const &machine_compute_specification,
    ParallelLayerGuidObliviousMachineMapping const &pre_mapping,
    ParallelLayerGuidObliviousMachineMapping const &post_mapping);
  
} // namespace FlexFlow

#endif
