#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_COPY_INSERTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_COPY_INSERTION_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_copy_info.dtg.h"
#include "task-spec/dynamic_graph/internal_dynamic_slot_site.dtg.h"

namespace FlexFlow {

bool node_is_copy(DynamicNodeAttrs const &n);
bool value_is_mapped(DynamicValueAttrs const &);

void require_node_is_ready_for_copy_insertion(DynamicNodeAttrs const &);
void require_value_is_ready_for_copy_insertion(DynamicValueAttrs const &);
void require_invocation_is_ready_for_copy_insertion(
    DynamicNodeInvocation const &);
void require_graph_is_ready_for_copy_insertion(
    DynamicOpenDataflowGraph const &);

void require_value_is_copy_inserted(DynamicValueAttrs const &);
void require_invocation_is_fully_copy_inserted(DynamicNodeInvocation const &);
void require_graph_is_fully_copy_inserted(DynamicOpenDataflowGraph const &);

std::map<DynamicTensorSlot, ParallelTensorMapping> get_mappings_for_invocation(
    DynamicNodeInvocation const &,
    std::map<InternalDynamicSlotSite, ParallelTensorMapping> const &);

DynamicNodeInvocation apply_mappings_for_invocation(
    dynamic_invocation_id_t const &,
    DynamicNodeInvocation const &,
    std::map<InternalDynamicSlotSite, ParallelTensorMapping> const &);

DynamicNodeInvocation make_copy_invocation(DynamicValueCopyInfo const &);

std::set<DynamicValueCopyInfo> copies_for_value(
    DynamicValueAttrs const &value_attrs,
    DynamicSlotSite const &src_site,
    std::set<InternalDynamicSlotSite> const &dst_sites,
    std::map<InternalDynamicSlotSite, ParallelTensorMapping> const &mappings);

std::set<DynamicValueCopyInfo> copies_for_internal_value(
    DynamicValueAttrs const &value_attrs,
    InternalDynamicSlotSite const &src_site,
    ParallelTensorMapping const &src_site_mapping,
    std::map<InternalDynamicSlotSite, ParallelTensorMapping> const
        &sink_site_mappings);

std::set<DynamicValueCopyInfo>
    infer_all_copies_in_graph(DynamicOpenDataflowGraph const &);

std::map<InternalDynamicSlotSite, ParallelTensorMapping>
    resolve_tensor_mappings(DynamicOpenDataflowGraph const &);

std::map<InternalDynamicSlotSite, ParallelTensorMapping>
    resolve_partial_tensor_mappings_from_node_mappings(
        DynamicOpenDataflowGraph const &);

std::map<InternalDynamicSlotSite, ParallelTensorMapping>
    resolve_missing_tensor_mappings_from_adjacent_values(
        DynamicOpenDataflowGraph const &,
        std::map<InternalDynamicSlotSite, ParallelTensorMapping> const &);

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
