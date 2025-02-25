#ifndef _FLEXFLOW_LOCAL_EXECUTION_UNALLOCATED_TENSORS_H
#define _FLEXFLOW_LOCAL_EXECUTION_UNALLOCATED_TENSORS_H

#include "local-execution/allocated_tensors.dtg.h"
#include "local-execution/gradient_tensor_source.h"
#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/unallocated_tensors.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/tensor_attrs.dtg.h"

namespace FlexFlow {

UnallocatedTensors generate_unallocated_tensors(
    AllocatedTensors const &,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &,
    GradientTensorSource &);

UnallocatedTensors generate_unallocated_tensors_with_optimizer(
    AllocatedTensors const &,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &,
    GradientTensorSource &,
    OptimizerTensorSource &,
    OptimizerAttrs const &);

} // namespace FlexFlow

#endif
