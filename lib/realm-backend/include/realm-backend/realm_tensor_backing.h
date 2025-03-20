
#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TENSOR_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "local-execution/allocated_tensors.dtg.h"
#include "local-execution/gradient_tensor_source.h"
#include "local-execution/loss_tensor_source.h"
#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/unallocated_tensors.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_task_argument_accessor.h"
#include "realm-backend/realm_tensor_backing.dtg.h"
#include "task-spec/lowered_tensor_t.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/tensor_role.dtg.h"
namespace FlexFlow {

  GenericTensorAccessorW get_tensor(RealmTensorBacking const &,
                                    TensorTypeVariant const &);
  
  std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
      get_tensor_backings(
          std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const &,
          std::unordered_map<TensorTypeVariant, TensorShape> const &,
          Allocator &);
  
  std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
      merge_optimizer_mappings(
          std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
              &allocated,
          std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
              &unallocated);
  
  RealmTensorBacking construct_realm_tensor_backing(AllocatedTensors const &,
                                                    UnallocatedTensors const &,
                                                    Allocator &);
  
  TensorSlotsBacking construct_tensor_slots_backing(RealmTensorBacking const &,
                                                    TaskBinding const &);
  
  } // namespace FlexFlow
  
  #endif