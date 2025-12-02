#include "local-execution/local_tensor_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/runtime_task_invocation/fwb_tensor_slot.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/is_submapeq_of.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalTensorBacking local_tensor_backing_for_tensor(
  symbolic_training_tensor_guid_t symbolic_tensor_guid) {

  return LocalTensorBacking{
    /*tensor_map=*/{
      {symbolic_tensor_guid, atomic_training_tensor_guid_t{0_n}},
    },
  };
}

AtomicTaskInvocation 
  lower_local_runtime_task_invocation_to_atomic_task_invocation(
    LocalTensorBacking const &,
    RuntimeTaskInvocation const &,
    RuntimeArgConfig const &) {
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
