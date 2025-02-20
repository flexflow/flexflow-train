#ifndef _FLEXFLOW_LOCAL_EXECUTION_ALLOCATED_TENSORS_H
#define _FLEXFLOW_LOCAL_EXECUTION_ALLOCATED_TENSORS_H

#include "realm-backend/allocated_tensors.dtg.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

bool are_allocated_forward_tensors_valid(
    AllocatedTensors const &,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &);
bool are_allocated_gradient_tensors_valid(
    AllocatedTensors const &,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &);
bool are_allocated_optimizer_tensors_valid(
    AllocatedTensors const &,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &);

bool are_allocated_tensors_valid(
    AllocatedTensors const &,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &);

bool is_allocated_tensor_backing_valid(
    TensorTypeVariant const &,
    std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const &,
    ArrayShape const &);

} // namespace FlexFlow

#endif
