#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H

#include <unordered_map>
#include "kernels/accessor.h"
#include "local-execution/computation_graph_training_tensor_ref_t.dtg.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph_from_cg_conversion.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

struct ComputationGraphInstance {
public:
  ComputationGraphInstance() = delete;

  explicit ComputationGraphInstance(
    TrainingSymbolicComputationGraphFromCgConversion const &,
    LocalTensorBacking const &,
    LocalTaskRegistry const &);
public:
  TrainingSymbolicComputationGraphFromCgConversion const &get_symbolic_training_graph_for_cg() const;
  LocalTensorBacking const &get_tensor_backing() const;  
  LocalTaskRegistry const &get_task_registry() const;
private:
  TrainingSymbolicComputationGraphFromCgConversion symbolic_training_graph_for_cg;
  LocalTensorBacking tensor_backing;
  LocalTaskRegistry task_registry;
};

ComputationGraphInstance create_computation_graph_instance(
  ComputationGraph const &,
  bidict<
    computation_graph_training_tensor_ref_t, 
    std::variant<GenericTensorAccessorW, GenericTensorAccessorR>
  > const &);

} // namespace FlexFlow

#endif
