#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

template <typename T>
static ParallelLayerAttrs make_layer_attrs(T const &op_attrs) {
  return ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{op_attrs},
      /*name=*/std::nullopt,
  };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("topological_ordering(ParallelComputationGraph)") {
    // TODO(@lockshaw) should probably be replaced with a rapidcheck test that
    // compares ParallelComputationGraph to DataflowGraph, but since we
    // currently don't have rapidcheck generation for DataflowGraph this will
    // have to do for now

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                16_p,
            },
        },
        DataType::FLOAT,
    };

    ElementUnaryAttrs relu_attrs = make_relu_attrs();

    ParallelLayerAddedResult layer1_added =
        pcg_add_input_layer(pcg, input_shape);
    parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
    parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

    ParallelLayerAddedResult layer2_added =
        add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
    parallel_layer_guid_t layer2 = layer2_added.parallel_layer;
    parallel_tensor_guid_t tensor2 = get_only(layer2_added.outputs);

    ParallelLayerAddedResult layer3_added =
        add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor2}, {});
    parallel_layer_guid_t layer3 = layer3_added.parallel_layer;
    parallel_tensor_guid_t tensor3 = get_only(layer3_added.outputs);

    std::vector<parallel_layer_guid_t> result = topological_ordering(pcg);
    std::vector<parallel_layer_guid_t> correct = {layer1, layer2, layer3};
    CHECK(result == correct);
  }

  TEST_CASE(
      "get_incoming_inputs(ParallelComputationGraph, parallel_layer_guid_t)") {
    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{10_p, 12_p},
        },
        DataType::FLOAT,
    };

    SUBCASE("layer has no inputs") {
      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_inputs(pcg, input_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      std::string my_op_name = "my op";

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      WeightAttrs bias_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_bias_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_projection =
          get_only(projection_weight_added.outputs);

      ParallelLayerAddedResult bias_weight_added =
          add_parallel_layer(pcg, make_layer_attrs(bias_weight_attrs), {}, {});
      parallel_tensor_guid_t t_bias = get_only(bias_weight_added.outputs);

      ParallelLayerAddedResult linear_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(linear_attrs),
                             {t_input},
                             {t_projection, t_bias});

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_inputs(pcg, linear_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {t_input};

      CHECK(result == correct);
    }
  }

  TEST_CASE(
      "get_source_layer(ParallelComputationGraph, parallel_tensor_guid_t)") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    ElementUnaryAttrs relu_attrs = make_relu_attrs();

    SUBCASE("single layer") {
      ParallelLayerAddedResult layer1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
      parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

      parallel_layer_guid_t result = get_source_layer(pcg, tensor1);
      parallel_layer_guid_t correct = layer1;
      CHECK(result == correct);
    }

    SUBCASE("two connected layers") {
      ParallelLayerAddedResult layer1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
      parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

      ParallelLayerAddedResult layer2_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
      parallel_layer_guid_t layer2 = layer2_added.parallel_layer;

      parallel_layer_guid_t result = get_source_layer(pcg, tensor1);
      parallel_layer_guid_t correct = layer1;
      CHECK(result == correct);
    }

    SUBCASE("three layers in series") {
      ParallelLayerAddedResult layer1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
      parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

      ParallelLayerAddedResult layer2_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
      parallel_layer_guid_t layer2 = layer2_added.parallel_layer;
      parallel_tensor_guid_t tensor2 = get_only(layer2_added.outputs);

      ParallelLayerAddedResult layer3_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
      parallel_layer_guid_t layer3 = layer3_added.parallel_layer;

      SUBCASE("tensor 1") {
        parallel_layer_guid_t result = get_source_layer(pcg, tensor1);
        parallel_layer_guid_t correct = layer1;
        CHECK(result == correct);
      }

      SUBCASE("tensor 2") {
        parallel_layer_guid_t result = get_source_layer(pcg, tensor2);
        parallel_layer_guid_t correct = layer2;
        CHECK(result == correct);
      }
    }
  }

  TEST_CASE(
      "get_incoming_weights(ParallelComputationGraph, parallel_layer_guid_t)") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    SUBCASE("layer has no inputs or weights") {
      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_weights(pcg, input_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult relu_added = add_parallel_layer(
          pcg, make_layer_attrs(make_relu_attrs()), {t_input}, {});

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_weights(pcg, relu_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights, and weights are separate by "
            "parallel ops") {
      std::string my_op_name = "my op";

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      RepartitionAttrs partition_input_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{0_n},
          /*repartition_degree=*/2_p,
      };

      ParallelLayerAddedResult partition_input_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_input}, {});
      parallel_tensor_guid_t t_partitioned_input =
          get_only(partition_input_added.outputs);

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_projection_weight =
          get_only(projection_weight_added.outputs);

      ReplicateAttrs replicate_projection_attrs = ReplicateAttrs{
          /*replicate_degree=*/2_p,
      };
      ParallelLayerAddedResult replicate_projection_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(replicate_projection_attrs),
                             {t_projection_weight},
                             {});
      parallel_tensor_guid_t t_replicated_projection_weight =
          get_only(replicate_projection_added.outputs);

      ParallelLayerAddedResult linear_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(linear_attrs),
                             {t_partitioned_input},
                             {t_replicated_projection_weight});

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_weights(pcg, linear_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {
          t_replicated_projection_weight};

      CHECK(result == correct);
    }
  }

  TEST_CASE("pcg_add_input_layer") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelComputationGraph result = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();
      pcg_add_input_layer(pcg, input_shape);
      return pcg;
    }();

    ParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
          /*op_attrs=*/PCGOperatorAttrs{InputAttrs{input_shape}},
          /*name=*/std::nullopt,
      };

      add_parallel_layer(/*pcg=*/pcg,
                         /*layer_attrs=*/layer_attrs,
                         /*inputs=*/{},
                         /*weights=*/{},
                         /*output_labels=*/std::vector{CreateGrad::NO});

      return pcg;
    }();

    CHECK(pcgs_are_isomorphic(result, correct));
  }
  
  TEST_CASE("pcg_get_mapping_along_edge") {
    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape par_input_shape = lift_to_parallel(input_shape);

    ParallelLayerAttrs partition_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            RepartitionAttrs{
                /*repartition_dim=*/ff_dim_t{0_n},
                /*repartition_degree=*/2_p,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelLayerAttrs relu_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            ElementUnaryAttrs{
                /*op_type=*/OperatorType::RELU,
                /*scalar=*/std::nullopt,
            },
        },
        /*name=*/std::nullopt,
    };

    SUBCASE("trivial mapping (relu into relu)") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input.outputs);
      ParallelLayerAddedResult partition_input =
          add_parallel_layer(pcg, partition_attrs, {t_input}, {});
      parallel_tensor_guid_t t_partition_input = get_only(partition_input.outputs);

      ParallelLayerAddedResult layer_1 =
          add_parallel_layer(pcg, relu_attrs, {t_partition_input}, {});
      parallel_tensor_guid_t t_layer_1 = get_only(layer_1.outputs);
      ParallelLayerAddedResult layer_2 =
          add_parallel_layer(pcg, relu_attrs, {t_layer_1}, {});

      ParallelComputationGraphEdge edge 
        = get_only(
            get_pcg_edges_from_layer_to_layer(
              /*pcg=*/pcg, 
              /*src=*/layer_1.parallel_layer, 
              /*dst=*/layer_2.parallel_layer));

      OperatorTaskSpaceToOperatorTaskSpaceMapping result = pcg_get_mapping_along_edge(pcg, edge);

      DimDomain<operator_task_space_dim_idx_t> layer_1_task_space = DimDomain<operator_task_space_dim_idx_t>{{
        {operator_task_space_dim_idx_t{0_n}, 2_p},
      }};

      DimDomain<operator_task_space_dim_idx_t> layer_2_task_space = layer_1_task_space;

      auto make_coord = [](nonnegative_int x) {
        return DimCoord{
          std::unordered_map<operator_task_space_dim_idx_t, nonnegative_int>{
            {operator_task_space_dim_idx_t{0_n}, x},
          },
        };
      };

      OperatorTaskSpaceToOperatorTaskSpaceMapping correct = OperatorTaskSpaceToOperatorTaskSpaceMapping{
        DimDomainMapping<operator_task_space_dim_idx_t, operator_task_space_dim_idx_t>{
          bidict<DimCoord<operator_task_space_dim_idx_t>, DimCoord<operator_task_space_dim_idx_t>>{
            {make_coord(0_n), make_coord(0_n)},
            {make_coord(1_n), make_coord(1_n)},
          },
          layer_1_task_space,
          layer_2_task_space,
        },
      };

      CHECK(result == correct); 
    }

    SUBCASE("nontrivial mapping (linear into linear)") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input.outputs);
      ParallelLayerAddedResult partition_input =
          add_parallel_layer(pcg, partition_attrs, {t_input}, {});
      parallel_tensor_guid_t t_partition_input = get_only(partition_input.outputs);

      ParallelLayerAttrs transpose_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
          TransposeAttrs{
            TensorDimPermutation{
              bidict<ff_dim_t, ff_dim_t>{
                {ff_dim_t{0_n}, ff_dim_t{1_n}},
                {ff_dim_t{1_n}, ff_dim_t{0_n}},
              },
            },
          },
        },
        /*name=*/std::nullopt,
      };

      ParallelLayerAddedResult layer_1 =
          add_parallel_layer(pcg, relu_attrs, {t_partition_input}, {});
      parallel_tensor_guid_t t_layer_1 = get_only(layer_1.outputs);
      ParallelLayerAddedResult layer_2 =
          add_parallel_layer(pcg, transpose_attrs, {t_layer_1}, {});

      ParallelComputationGraphEdge edge 
        = get_only(
            get_pcg_edges_from_layer_to_layer(
              /*pcg=*/pcg, 
              /*src=*/layer_1.parallel_layer, 
              /*dst=*/layer_2.parallel_layer));

      OperatorTaskSpaceToOperatorTaskSpaceMapping result = pcg_get_mapping_along_edge(pcg, edge);

      DimDomain<operator_task_space_dim_idx_t> layer_1_task_space = DimDomain<operator_task_space_dim_idx_t>{{
        {operator_task_space_dim_idx_t{0_n}, 2_p},
      }};

      DimDomain<operator_task_space_dim_idx_t> layer_2_task_space = layer_1_task_space;

      auto make_coord = [](nonnegative_int x) {
        return DimCoord{
          std::unordered_map<operator_task_space_dim_idx_t, nonnegative_int>{
            {operator_task_space_dim_idx_t{0_n}, x},
          },
        };
      };

      OperatorTaskSpaceToOperatorTaskSpaceMapping correct = OperatorTaskSpaceToOperatorTaskSpaceMapping{
        DimDomainMapping<operator_task_space_dim_idx_t, operator_task_space_dim_idx_t>{
          bidict<DimCoord<operator_task_space_dim_idx_t>, DimCoord<operator_task_space_dim_idx_t>>{
            {make_coord(0_n), make_coord(1_n)},
            {make_coord(1_n), make_coord(0_n)},
          },
          layer_1_task_space,
          layer_2_task_space,
        },
      };
    }
  }
}
