#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
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
#include <doctest/doctest.h>

using namespace ::FlexFlow;
using namespace Realm;

bool did_loss_decrease(GenericTensorAccessorW const &first_epoch, GenericTensorAccessorW const & last_epoch) {
  float* first_epoch_ptr = first_epoch.get_float_ptr();
  float* last_epoch_ptr = last_epoch.get_float_ptr();
  
  int batch_size = first_epoch.shape.at(ff_dim_t{nonnegative_int{0}}).unwrap_nonnegative();
  for (int i = 0; i < batch_size; i++) {
    if (first_epoch_ptr[i] < last_epoch_ptr[i]) {
      return false;
    }
  }

  return true;
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Realm::Processor p) {
    // initialize runtime
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle{};
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
    loss_tensor_t label_tensor =
        loss_tensor_source.new_loss_tensor();

    nonnegative_int batch_size = 10_n;
    nonnegative_int data_dim = 16_n;
    nonnegative_int output_dim = 32_n;

    TensorShape output_tensor_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, output_dim}},
        DataType::FLOAT};

    GenericTensorAccessorW label_tensor_backing =
        allocators[0].allocate_tensor(output_tensor_shape);
    AllocatedTensors allocated_tensors = AllocatedTensors{
        {
         {TensorTypeVariant{label_tensor},
         label_tensor_backing}},
        {},
        {}};

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, data_dim}},
        DataType::FLOAT};

    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{data_dim, output_dim}},
        DataType::FLOAT};

    LayerAddedResult inputs_layer =
        add_input_layer(computation_graph, input_tensor_shape);

    LayerAddedResult weights_layer = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape, InitializerAttrs{GlorotNormalAttrs{0}}}},
                   std::nullopt},
        {},
        {});

    LayerAddedResult linear_operator = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        inputs_layer.outputs,
        weights_layer.outputs);
    tensor_guid_t logit_tensor = get_only(linear_operator.outputs);

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

      int num_epochs = 10;
      std::vector<GenericTensorAccessorW> loss_values (num_epochs);

      for (int i = 0; i < num_epochs; i++) {
        model_training_instance.forward();
        model_training_instance.backward();
        model_training_instance.update();
        loss_values[i] = model_training_instance.get_loss_tensor_backing();
      }
      // Assert that each sample in the batch has a lower loss in last epoch than the first epoch
      CHECK(did_loss_decrease(loss_values[0], loss_values[num_epochs - 1]));
      printf("passed\n");
    }
  }
}