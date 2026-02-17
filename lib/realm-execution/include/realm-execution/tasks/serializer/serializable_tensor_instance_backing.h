#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_TENSOR_INSTANCE_BACKING_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_TENSOR_INSTANCE_BACKING_H

#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.dtg.h"
#include "realm-execution/tensor_instance_backing.dtg.h"

namespace FlexFlow {

SerializableTensorInstanceBacking
    tensor_instance_backing_to_serializable(TensorInstanceBacking const &);
TensorInstanceBacking tensor_instance_backing_from_serializable(
    SerializableTensorInstanceBacking const &);

} // namespace FlexFlow

#endif
