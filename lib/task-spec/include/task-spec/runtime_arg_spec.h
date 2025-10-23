#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_SPEC_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_SPEC_H

#include "task-spec/op_arg_spec.dtg.h"
#include "task-spec/runtime_arg_spec.dtg.h"
#include "task-spec/symbolic_layer_tensor_shape_signature.dtg.h"

namespace FlexFlow {

std::type_index get_type_index(RuntimeArgSpec const &);

RuntimeArgSpec lower_op_arg_spec_to_runtime_arg_spec(
    OpArgSpec const &op_arg_spec,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTensorShapeSignature const &op_shape_signature);

RuntimeArgSpec lower_op_arg_ref_spec_to_runtime_arg_spec(
    OpArgRefSpec const &,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTensorShapeSignature const &);

}

#endif
