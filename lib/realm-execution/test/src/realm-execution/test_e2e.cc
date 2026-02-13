#include "internal/realm_test_utils.h"
#include "kernels/allocation.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "realm-execution/distributed_device_handle.h"
#include "realm-execution/pcg_instance/pcg_instance.h"
#include "realm-execution/realm_manager.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

namespace test {

using namespace ::FlexFlow;
namespace Realm = ::FlexFlow::Realm;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/0_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager(&fake_argc, &fake_argv);

    (void)manager.start_controller([](RealmContext &ctx) {
      Allocator allocator = ctx.get_current_device_allocator();

      positive_int batch_size = 10_p;
      positive_int data_dim = 16_p;
      positive_int hidden_dim = 32_p;
      positive_int output_dim = 1_p;

      TensorShape output_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

      GenericTensorAccessorW label_tensor_backing =
          allocator.allocate_tensor(output_tensor_shape);

      // construct computation graph
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      TensorShape input_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

      TensorShape label_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};
      GenericTensorAccessorW label_tensor =
          allocator.allocate_tensor(label_tensor_shape);

      TensorShape weight_shape_1 = TensorShape{
          TensorDims{FFOrdered{hidden_dim, data_dim}}, DataType::FLOAT};
      TensorShape weight_shape_2 = TensorShape{
          TensorDims{FFOrdered{output_dim, hidden_dim}}, DataType::FLOAT};

      ParallelLayerAddedResult inputs_layer =
          pcg_add_input_layer_with_grad(pcg, input_tensor_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult weights_layer_1 = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{
              PCGOperatorAttrs{WeightAttrs{
                  weight_shape_1, InitializerAttrs{GlorotNormalAttrs{0}}}},
              std::nullopt},
          {},
          {});
      parallel_tensor_guid_t t_weights_1 =
          require_only_key(weights_layer_1.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult weights_layer_2 = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{
              PCGOperatorAttrs{WeightAttrs{
                  weight_shape_2, InitializerAttrs{GlorotNormalAttrs{0}}}},
              std::nullopt},
          {},
          {});
      parallel_tensor_guid_t t_weights_2 =
          require_only_key(weights_layer_2.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult linear_operator_1 = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{hidden_dim,
                                                          /*use_bias=*/false,
                                                          DataType::FLOAT,
                                                          Activation::RELU,
                                                          std::nullopt}},
                             std::nullopt},
          {
              {
                  TensorSlotName::INPUT,
                  t_input,
              },
          },
          {
              {
                  TensorSlotName::WEIGHT,
                  t_weights_1,
              },
          });
      parallel_tensor_guid_t t_linear_1 =
          require_only_key(linear_operator_1.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult linear_operator_2 = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{output_dim,
                                                          /*use_bias=*/false,
                                                          DataType::FLOAT,
                                                          Activation::RELU,
                                                          std::nullopt}},
                             std::nullopt},
          {
              {
                  TensorSlotName::INPUT,
                  t_linear_1,
              },
          },
          {
              {
                  TensorSlotName::WEIGHT,
                  t_weights_2,
              },
          });
      parallel_tensor_guid_t t_linear_2 =
          require_only_key(linear_operator_2.outputs, TensorSlotName::OUTPUT);

      MappedParallelComputationGraph mpcg{pcg, {}};

      // instantiate computation graph
      LossAttrs loss_attrs = LossAttrs{
          NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
      OptimizerAttrs optimizer_attrs =
          OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                           /*momentum=*/0.9,
                                           /*nesterov=*/false,
                                           /*weight_decay=*/0.001}};

      std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
          input_tensors;

      DistributedDeviceHandle device_handle = create_distributed_device_handle(
          ctx,
          /*workSpaceSize=*/1024 * 1024,
          /*allowTensorOpMathConversion=*/true);

      PCGInstance pcg_instance = create_pcg_instance(
          /*ctx=*/ctx,
          /*mpcg=*/mpcg,
          /*optimizer=*/optimizer_attrs,
          /*loss=*/loss_attrs,
          /*label_tensor=*/label_tensor,
          /*logit_tensor=*/t_linear_2,
          /*input_tensors=*/input_tensors,
          /*profiling_settings=*/ProfilingSettings{0, 0},
          /*device_handle=*/device_handle,
          /*iteration_config=*/FFIterationConfig{1_p});

      // begin training loop
      int num_epochs = 5;
      std::vector<GenericTensorAccessorR> loss_values;

      for (int i = 0; i < num_epochs; i++) {
        perform_all_passes_for_pcg_instance(
            /*instance=*/pcg_instance,
            /*profiling_settings=*/ProfilingSettings{0, 0},
            /*device_handle=*/device_handle,
            /*iteration_config=*/FFIterationConfig{1_p});
        // loss_values.push_back(copy_tensor_accessor_r(
        //     pcg_instance.get_loss_tensor_accessor().value(),
        //     allocator));
      }

      // // Assert that each sample in the batch has a lower loss in last epoch
      // // than the first epoch
      // GenericTensorAccessorR first_epoch_loss = loss_values.at(0);
      // GenericTensorAccessorR last_epoch_loss = loss_values.back();
      // CHECK_MESSAGE(did_loss_decrease(first_epoch_loss, last_epoch_loss),
      //               check_kv("first_epoch_loss",
      //                        format_accessor_r_contents(first_epoch_loss)),
      //               check_kv("last_epoch_loss",
      //                        format_accessor_r_contents(last_epoch_loss)));
    });
  }
}

} // namespace test
