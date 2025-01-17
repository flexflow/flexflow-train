#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_NETWORK_COST_MODEL_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_NETWORK_COST_MODEL_H

#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "pcg/machine_specification.dtg.h"

namespace FlexFlow {

float estimate_communication_cost(MachineSpecification const &, 
                                  TensorSetMovement const &);

} // namespace FlexFlow

#endif
