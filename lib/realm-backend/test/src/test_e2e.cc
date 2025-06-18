#include "kernels/compare_tensor_accessors.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/tensor_accessor_reductions.h"
#include "kernels/test_utils.h"
#include "local-execution/allocated_tensors.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "test_utils.h"
#include "utils/containers/get_only.h"
#include "realm-backend/model_training_instance.h"

using namespace ::FlexFlow;
using namespace Realm;

bool did_loss_decrease(GenericTensorAccessorR const &first_epoch,
                       GenericTensorAccessorR const &last_epoch) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  return tensor_accessor_all(
      compare_tensor_accessors_le(last_epoch, first_epoch, cpu_allocator));
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Realm::Processor p) {
  // initialize runtime
  ManagedFFStream managed_stream{};
  ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
      /*workSpaceSize=*/1024 * 1024,
      /*allowTensorOpMathConversion=*/true);
  std::vector<Processor> worker_procs;
  std::vector<Allocator> allocators;
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine())
                                  .only_kind(Processor::TOC_PROC);
  assert(pq.count() > 0);
  for (Processor p : pq) {
    worker_procs.push_back(p);
    allocators.push_back(create_realm_memory_allocator(p));
  }

  // allocate label tensors
  LossTensorSource loss_tensor_source;
  loss_tensor_t label_tensor = loss_tensor_source.new_loss_tensor();

  positive_int batch_size = 10_p;
  positive_int data_dim = 16_p;
  positive_int hidden_dim = 32_p;
  positive_int output_dim = 1_p;

  TensorShape input_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};
  TensorShape output_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  GenericTensorAccessorW label_tensor_backing = create_random_filled_accessor_w(
      output_tensor_shape, allocators[0]);

  // construct computation graph
  ComputationGraph computation_graph = make_empty_computation_graph();

  TensorShape weight_shape_1 = TensorShape{
      TensorDims{FFOrdered{data_dim, hidden_dim}}, DataType::FLOAT};
  TensorShape weight_shape_2 = TensorShape{
      TensorDims{FFOrdered{hidden_dim, output_dim}}, DataType::FLOAT};

  GenericTensorAccessorW weight_1_backing = create_random_filled_accessor_w(
      weight_shape_1, allocators[0]);
  GenericTensorAccessorW weight_2_backing = create_random_filled_accessor_w(
      weight_shape_2, allocators[0]);

  LayerAddedResult inputs_layer =
      add_input_layer_with_grad(computation_graph, input_tensor_shape);
  tensor_guid_t input_tensor_guid = get_only(inputs_layer.outputs);
  GenericTensorAccessorW input_tensor_backing = create_random_filled_accessor_w(
      input_tensor_shape, allocators[0]);

  LayerAddedResult weights_layer_1 = add_layer(
      computation_graph,
      LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                      weight_shape_1, InitializerAttrs{GlorotNormalAttrs{0}}}},
                  std::nullopt},
      {},
      {});
  tensor_guid_t weight_1_tensor_guid = get_only(weights_layer_1.outputs);

  LayerAddedResult weights_layer_2 = add_layer(
      computation_graph,
      LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                      weight_shape_2, InitializerAttrs{GlorotNormalAttrs{0}}}},
                  std::nullopt},
      {},
      {});
  tensor_guid_t weight_2_tensor_guid = get_only(weights_layer_2.outputs);

  LayerAddedResult linear_operator_1 = add_layer(
      computation_graph,
      LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{hidden_dim,
                                                      /*use_bias=*/false,
                                                      DataType::FLOAT,
                                                      std::nullopt,
                                                      std::nullopt}},
                  std::nullopt},
      inputs_layer.outputs,
      weights_layer_1.outputs);

  LayerAddedResult linear_operator_2 = add_layer(
      computation_graph,
      LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                      /*use_bias=*/false,
                                                      DataType::FLOAT,
                                                      std::nullopt,
                                                      std::nullopt}},
                  std::nullopt},
      linear_operator_1.outputs,
      weights_layer_2.outputs);

  tensor_guid_t logit_tensor = get_only(linear_operator_2.outputs);

  RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
      DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
      EnableProfiling::YES,
      ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

  // initialize training backing
  LossAttrs loss_attrs = LossAttrs{NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
  OptimizerAttrs optimizer_attrs =
      OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                        /*momentum=*/0.9,
                                        /*nesterov=*/false,
                                        /*weight_decay=*/0.001}};


  GradientTensorSource gradient_tensor_source;
  OptimizerTensorSource optimizer_tensor_source;

  AllocatedTensors allocated_tensors = AllocatedTensors{
      /*tensor_type_backings=*/{
          {TensorTypeVariant{label_tensor}, label_tensor_backing},
          {TensorTypeVariant{input_tensor_guid}, input_tensor_backing},
          {TensorTypeVariant{weight_1_tensor_guid}, weight_1_backing},
          {TensorTypeVariant{weight_2_tensor_guid}, weight_2_backing},
      },
      /*gradient_mapping=*/{},
      /*optimizer_mapping*/ {},
  };

  {
    printf("\nRunning test %d: E2ETest...\n", 1);
    RealmTrainingBacking realm_training_backing = RealmTrainingBacking(
        p, worker_procs, allocators, allocated_tensors, gradient_tensor_source,
        optimizer_tensor_source, computation_graph, runtime_arg_config,
        optimizer_attrs);
    // begin training loop                      
    ModelTrainingInstance model_training_instance = ModelTrainingInstance{
      realm_training_backing, logit_tensor, label_tensor, loss_attrs, optimizer_attrs
    };

    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    int num_epochs = 5;
    std::vector<GenericTensorAccessorR> loss_values;

    for (int i = 0; i < num_epochs; i++) {
      model_training_instance.forward();
      model_training_instance.backward();
      model_training_instance.update();
      loss_values.push_back(copy_tensor_accessor_r(
          model_training_instance.get_loss_tensor_accessor(), cpu_allocator));
    }

    // Assert that each sample in the batch has a lower loss in last epoch than
    // the first epoch
    GenericTensorAccessorR first_epoch_loss = loss_values.at(0);
    
    GenericTensorAccessorR last_epoch = loss_values.back();

    assert(did_loss_decrease(first_epoch_loss, last_epoch));
    printf("passed\n");
  }
}
