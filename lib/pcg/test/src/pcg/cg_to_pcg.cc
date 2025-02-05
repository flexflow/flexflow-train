#include "pcg/cg_to_pcg.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cg_to_pcg") {
    std::string input_name = "input";
    std::string dense_name = "dense";
    std::string relu_name = "relu";

    ComputationGraph cg = [&] {
      ComputationGraph cg = make_empty_computation_graph();

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{10_n, 12_n}}, DataType::FLOAT};
      TensorAttrs input_attrs =
          TensorAttrs{input_shape, std::nullopt, std::nullopt, CreateGrad::YES};
      LayerAttrs input_layer_attrs =
          LayerAttrs{ComputationGraphOpAttrs{InputAttrs{}}, input_name};
      LayerAddedResult input_added =
          add_layer(cg, input_layer_attrs, {}, {input_attrs});
      tensor_guid_t input_tensor = get_only(input_added.outputs);

      LinearAttrs linear_attrs = LinearAttrs{/*out_channels=*/8_n,
                                             /*use_bias=*/true,
                                             /*data_type=*/DataType::FLOAT,
                                             /*activation=*/Activation::RELU,
                                             /*regularizer=*/std::nullopt};
      TensorShape dense_output_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{10_n, 8_n}}, DataType::FLOAT};
      LayerAttrs dense_layer_attrs =
          LayerAttrs{ComputationGraphOpAttrs{linear_attrs}, dense_name};
      LayerAddedResult dense_added = add_layer(cg,
                                               dense_layer_attrs,
                                               {input_tensor},
                                               {TensorAttrs{dense_output_shape,
                                                            std::nullopt,
                                                            std::nullopt,
                                                            CreateGrad::YES}});
      tensor_guid_t dense_output = get_only(dense_added.outputs);

      ElementUnaryAttrs relu_attrs =
          ElementUnaryAttrs{OperatorType::RELU, std::nullopt};
      LayerAttrs relu_layer_attrs =
          LayerAttrs{ComputationGraphOpAttrs{relu_attrs}, relu_name};
      add_layer(cg,
                relu_layer_attrs,
                {dense_output},
                {TensorAttrs{dense_output_shape,
                             std::nullopt,
                             std::nullopt,
                             CreateGrad::YES}});

      return cg;
    }();

    ParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelTensorShape input_shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{ShardParallelDim{10_n, 1_n},
                                          ShardParallelDim{12_n, 1_n}},
              ReplicaParallelDimSet{SumDegree{1_n}, DiscardCopyDegree{1_n}}},
          DataType::FLOAT};

      ParallelLayerAttrs input_layer_attrs =
          ParallelLayerAttrs{PCGOperatorAttrs{InputAttrs{}}, input_name};

      ParallelLayerAddedResult input_added = add_parallel_layer(
          pcg,
          input_layer_attrs,
          {},
          {ParallelTensorAttrs{
              input_shape, std::nullopt, std::nullopt, CreateGrad::YES}});

      parallel_tensor_guid_t input_tensor = get_only(input_added.outputs);

      LinearAttrs linear_attrs = LinearAttrs{/*out_channels=*/8_n,
                                             /*use_bias=*/true,
                                             /*data_type=*/DataType::FLOAT,
                                             /*activation=*/Activation::RELU,
                                             /*regularizer=*/std::nullopt};

      ParallelLayerAttrs dense_layer_attrs =
          ParallelLayerAttrs{PCGOperatorAttrs{linear_attrs}, dense_name};

      ParallelTensorShape dense_output_shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{ShardParallelDim{10_n, 1_n},
                                          ShardParallelDim{8_n, 1_n}},
              ReplicaParallelDimSet{SumDegree{1_n}, DiscardCopyDegree{1_n}}},
          DataType::FLOAT};

      ParallelLayerAddedResult dense_added =
          add_parallel_layer(pcg,
                             dense_layer_attrs,
                             {input_tensor},
                             {ParallelTensorAttrs{dense_output_shape,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  CreateGrad::YES}});

      parallel_tensor_guid_t dense_output = get_only(dense_added.outputs);

      ElementUnaryAttrs relu_attrs =
          ElementUnaryAttrs{OperatorType::RELU, std::nullopt};
      ParallelLayerAttrs relu_layer_attrs =
          ParallelLayerAttrs{PCGOperatorAttrs{relu_attrs}, relu_name};

      add_parallel_layer(pcg,
                         relu_layer_attrs,
                         {dense_output},
                         {ParallelTensorAttrs{dense_output_shape,
                                              std::nullopt,
                                              std::nullopt,
                                              CreateGrad::YES}});
      return pcg;
    }();

    ParallelComputationGraph result = cg_to_pcg(cg);

    CHECK(pcgs_are_isomorphic(result, correct));
  }
}
