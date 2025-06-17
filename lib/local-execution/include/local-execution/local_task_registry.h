#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "local-execution/local_task_registry.dtg.h"
#include "pcg/layer_attrs.dtg.h"
#include "task-spec/op_task_type.dtg.h"
#include "utils/units/milliseconds_t.dtg.h"

namespace FlexFlow {

LocalTaskRegistry construct_local_task_registry_for_layers(
    std::unordered_map<layer_guid_t, LayerAttrs> const &);

bool registry_contains_task_for_layer(LocalTaskRegistry const &,
                                      layer_guid_t const &,
                                      OpTaskType const &);

std::optional<milliseconds_t> call_task_impl(LocalTaskRegistry const &,
                                             task_id_t const &task_id,
                                             TaskArgumentAccessor const &acc);

} // namespace FlexFlow

#endif
