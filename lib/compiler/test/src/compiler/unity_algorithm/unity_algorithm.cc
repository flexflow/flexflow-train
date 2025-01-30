#include "compiler/unity_algorithm/unity_algorithm.h"
#include "../cost_estimator_for_test.h"
#include "doctest/doctest.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/replica_type.dtg.h"
#include "op-attrs/shard_parallel_dim.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/integer_conversions.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("graph_optimize") {
    // TODO: recover this by implementing
    // parallel_computation_graph_from_computation_graph ComputationGraph cg =
    // [&] {
    //   ComputationGraphBuilder b;
    //   TensorShape input_tensor_shape = TensorShape{
    //     TensorDims{
    //       FFOrdered<size_t> {32, 64},
    //     },
    //     DataType::FLOAT,
    //   };
    //   tensor_guid_t t = b.create_input(input_tensor_shape, CreateGrad::YES);
    //   t = b.dense(t,
    //               /*outDim=*/16,
    //               /*activation=*/std::nullopt);
    //   t = b.gelu(t);
    //   t = b.dense(t,
    //               /*outDim=*/12,
    //               /*activation=*/std::nullopt,
    //               /*use_bias=*/false,
    //               /*data_type=*/DataType::FLOAT,
    //               /*kernel_initializer=*/std::nullopt,
    //               /*bias_initializer=*/std::nullopt);
    //   t = b.relu(t);
    //   t = b.dense(t,
    //               /*outDim=*/8,
    //               /*activation=*/Activation::RELU);
    //   return b.computation_graph;
    // }();

    // ParallelComputationGraph pcg =
    // parallel_computation_graph_from_computation_graph(cg);

    ParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      int in_channels = 24;
      int batch_size = 4;
      int batch_degree = 2;
      parallel_tensor_guid_t t = b.create_input_tensor(ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{size_t_from_int(batch_size), batch_degree},
                  ShardParallelDim{size_t_from_int(in_channels), 1},
              },
              ReplicaParallelDimSet{
                  SumDegree{1},
                  DiscardCopyDegree{1},
              },
          },
          DataType::FLOAT,
      });
      t = b.dense(t,
                  /*outDim=*/16,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt);
      t = b.relu(t);
      t = b.dense(t,
                  /*outDim=*/8,
                  /*activation=*/Activation::RELU);

      return b.pcg;
    }();

    CostEstimator cost_estimator = make_fake_cost_estimator(
        [](OpCostEstimateKey const &k) {
          return OpCostMetrics{
              /*forward_runtime=*/1.0,
              /*backward_runtime=*/2.0,
              /*memory=*/nonnegative_int{1},
          };
        },
        [](TensorSetMovement const &) { return 1.0; });

    MachineSpecification full_machine_spec = MachineSpecification{
        /*num_nodes=*/2,
        /*num_cpus_per_node=*/1,
        /*num_gpus_per_node=*/1,
        /*inter_node_bandwidth=*/1,
        /*intra_node_bandwidth=*/1,
    };

    // TODO: set up substitutions
    std::vector<Substitution> substitutions = {};

    UnitySearchConfig search_config = UnitySearchConfig{
        /*alpha=*/1.0,
        /*budget=*/20,
        /*threshold=*/1000.0,
        /*max_num_ops=*/100,
    };

    // SearchResult result = graph_optimize(pcg, cost_estimator,
    // full_machine_spec, substitutions, search_config);

    // TODO: check the result
  }
}
