#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ELEMENT_BINARY_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ELEMENT_BINARY_H

#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"
#include "task-spec/task_signature_impl.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(ElementBinaryAttrs const &);

OpTaskInvocation init(ElementBinaryAttrs const &);
OpTaskInvocation forward(ElementBinaryAttrs const &);
OpTaskInvocation backward(ElementBinaryAttrs const &);

TaskImplFunction get_element_binary_init_task_impl();
TaskImplFunction get_element_binary_fwd_task_impl();
TaskImplFunction get_element_binary_bwd_task_impl();

OpTaskSignature get_element_binary_init_signature();
OpTaskSignature get_element_binary_fwd_signature();
OpTaskSignature get_element_binary_bwd_signature();

} // namespace FlexFlow

#endif
