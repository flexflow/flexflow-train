#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H

#include "kernels/accessor.h"
#include "local-execution/local_task_registry.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include <unordered_map>

namespace FlexFlow {

struct ComputationGraphInstance {
public:
  ComputationGraphInstance() = delete;

  explicit ComputationGraphInstance(DynamicOpenDataflowGraph const &,
                                    LocalTaskRegistry const &);

public:
  DynamicOpenDataflowGraph const &get_expanded_dataflow_graph() const;
  LocalTaskRegistry const &get_task_registry() const;

private:
  DynamicOpenDataflowGraph expanded_dataflow_graph;
  LocalTaskRegistry task_registry;
};

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &,
    bidict<tensor_guid_t, // FIXME (Elliott): not sure this is correct
           std::variant<GenericTensorAccessorW, GenericTensorAccessorR>> const
        &,
    LocalTaskRegistry task_registry);

} // namespace FlexFlow

#endif
