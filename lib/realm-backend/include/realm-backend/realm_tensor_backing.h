
#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TENSOR_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "realm-backend/realm_task_argument_accessor.h"
#include "realm-backend/realm_allocator.h"
#include "local-execution/task_invocation.dtg.h"
#include "local-execution/tensor_role.dtg.h"
#include "local-execution/lowered_tensor_t.dtg.h"
#include "local-execution/lowered_tensor_source.h"
#include "local-execution/optimizer_tensor_t.dtg.h"
#include "local-execution/loss_tensor_t.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "pcg/layer_guid_t.dtg.h"

namespace FlexFlow {

using TensorRegionMap =
    std::unordered_map<lowered_tensor_t, RealmRegion>;
using TensorShapeMap =
    std::unordered_map<lowered_tensor_t, TensorShape>;

struct RealmTensorBacking {
  RealmTensorBacking();

public:
  void allocate_layer_tensors(layer_guid_t const &,
                              ComputationGraph const &,
                              RealmAllocator &);
  void allocate_tensors_by_role(TensorRole const &,
                                layer_guid_t const &,
                                ComputationGraph const &,
                                RealmAllocator &);
  void allocate_optimizer_tensors(tensor_guid_t const &,
                                  std::vector<optimizer_tensor_t> const &,
                                  RealmAllocator &);
  TensorSlotsBacking
      construct_tensor_slots_backing(TaskBinding const &) const;

  GenericTensorAccessorW const &
      get_tensor_backing(lowered_tensor_t const &) const;

  bool is_tensor_allocated(lowered_tensor_t const &) const;

public:
  // tensors
  TensorRegionMap tensor_regions;
  TensorShapeMap tensor_shapes;
  std::unordered_map<tensor_guid_t, lowered_tensor_t> tensor_lowering_mapping;
  std::unordered_map<tensor_guid_t, lowered_tensor_t> gradient_tensor_lowering_mapping;
  std::unordered_map<optimizer_tensor_t, lowered_tensor_t> optimizer_tensor_lowering_mapping;
  std::unordered_map<loss_tensor_t, lowered_tensor_t> loss_tensor_lowering_mapping;
  LoweredTensorSource lowered_tensor_source;
};

} // namespace FlexFlow

#endif
