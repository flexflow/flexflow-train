#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/allocated_tensors.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_training_backing.h"
#include "test_utils.h"

using namespace ::FlexFlow;
using namespace Realm;

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Realm::Processor p) {
  // initialize runtime configs
  ManagedFFStream managed_stream{};
  ManagedPerDeviceFFHandle managed_handle{};
  std::vector<Processor> worker_procs;
  std::vector<Allocator> allocators;
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine())
                                   .only_kind(Processor::TOC_PROC);
  for (Processor p : pq) {
    worker_procs.push_back(p);
    allocators.push_back(create_realm_memory_allocator(p));
  }

  AllocatedTensors allocated_tensors = make_empty_allocated_tensors();

  // construct computation graph
  ComputationGraph computation_graph = make_empty_computation_graph();

  nonnegative_int batch_size = 10_n;
  nonnegative_int data_dim = 16_n;
  nonnegative_int output_dim = 32_n;

  TensorShape input_tensor_shape =
      TensorShape{TensorDims{FFOrdered<nonnegative_int>{batch_size, data_dim}},
                  DataType::FLOAT};

  TensorShape weight_shape =
      TensorShape{TensorDims{FFOrdered<nonnegative_int>{data_dim, output_dim}},
                  DataType::FLOAT};

  LayerAddedResult inputs_layer =
      add_input_layer(computation_graph, input_tensor_shape);

  LayerAddedResult weights_layer = add_layer(
      computation_graph,
      LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                     weight_shape, InitializerAttrs{ZeroInitializerAttrs{}}}},
                 "weights"},
      {}, {});

  LayerAddedResult linear_operator =
      add_layer(computation_graph,
                LayerAttrs{ComputationGraphOpAttrs{
                               LinearAttrs{output_dim,
                                           /*use_bias=*/false, DataType::FLOAT,
                                           Activation::RELU, std::nullopt}},
                           "linear"},
                inputs_layer.outputs, weights_layer.outputs);

  RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
      DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
      EnableProfiling::YES,
      ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

  GradientTensorSource gradient_tensor_source;
  OptimizerTensorSource optimizer_tensor_source;

  int test_id = 0;

  {
    printf("Running test %d: SGDOptimizerAttrs, momentum=0\n", ++test_id);
    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                        /*momentum=*/0.0f,
                                        /*nesterov=*/false,
                                        /*weight_decay=*/0.001}};
    RealmTrainingBacking realm_training_backing = RealmTrainingBacking(
        p, worker_procs, allocators, allocated_tensors, gradient_tensor_source,
        optimizer_tensor_source, computation_graph, runtime_arg_config,
        optimizer_attrs);
    execute_update(realm_training_backing, linear_operator.layer, optimizer_attrs);
  }

  {
    printf("Running test %d: SGDOptimizerAttrs, momentum=0.9\n", ++test_id);
    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                        /*momentum=*/0.9,
                                        /*nesterov=*/false,
                                        /*weight_decay=*/0.001}};
    RealmTrainingBacking realm_training_backing = RealmTrainingBacking(
        p, worker_procs, allocators, allocated_tensors, gradient_tensor_source,
        optimizer_tensor_source, computation_graph, runtime_arg_config,
        optimizer_attrs);
    execute_update(realm_training_backing, linear_operator.layer, optimizer_attrs);
  }
  
  {
    printf("Running test %d: AdamOptimizerAttrs\n", ++test_id);
    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{AdamOptimizerAttrs{/*alpha=*/0.001,
                                        /*beta1=*/0.9,
                                        /*beta2=*/0.999,
                                        /*weight_decay=*/0.001,
                                        /*alpha_t=*/0.001,
                                        /*beta_t=*/0.9,
                                        /*beta2_t=*/0.999,
                                        /*epsilon=*/1e-8}};
    RealmTrainingBacking realm_training_backing = RealmTrainingBacking(
        p, worker_procs, allocators, allocated_tensors, gradient_tensor_source,
        optimizer_tensor_source, computation_graph, runtime_arg_config,
        optimizer_attrs);
    execute_update(realm_training_backing, linear_operator.layer, optimizer_attrs);
  }
}
