#include "compiler/mcmc/mcmc_over_mapped_pcg.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "doctest/doctest.h"
#include "internal/runtime_only_cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/replica_type.dtg.h"
#include "op-attrs/shard_parallel_dim.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "utils/integer_conversions.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mcmc_graph_optimize") {
    ComputationGraph cg = [&] {
      ComputationGraphBuilder b;
      TensorShape input_tensor_shape = TensorShape{
          TensorDims{
              FFOrdered{32_p, 64_p},
          },
          DataType::FLOAT,
      };
      tensor_guid_t t = b.create_input(input_tensor_shape, CreateGrad::YES);
      t = b.dense(t,
                  /*outDim=*/16_p,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12_p,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt);
      t = b.relu(t);
      t = b.dense(t,
                  /*outDim=*/8_p,
                  /*activation=*/Activation::RELU);
      return b.computation_graph;
    }();

    ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

    RuntimeOnlyCostEstimator cost_estimator =
        make_fake_constant_runtime_only_cost_estimator(
            /*forward_op_cost=*/10_ms,
            /*backward_op_cost=*/10_ms,
            /*comm_cost=*/1_ms);

    MachineSpecification full_machine_spec = MachineSpecification{
        /*num_nodes=*/2_p,
        /*num_cpus_per_node=*/1_p,
        /*num_gpus_per_node=*/1_p,
        /*inter_node_bandwidth=*/1,
        /*intra_node_bandwidth=*/1,
    };

    MCMCOverMappedPCGConfig no_search =
        MCMCOverMappedPCGConfig{/*temperature=*/1.0,
                                /*num_iterations=*/1_n,
                                /*substitution_interval=*/5_n,
                                /*device_type=*/DeviceType::GPU};

    SearchResult base_result =
        mcmc_graph_optimize(pcg, cost_estimator, full_machine_spec, no_search);
    float base_runtime =
        task_simulator_estimate_forward_pass_time(base_result.pcg,
                                                  cost_estimator,
                                                  base_result.machine_mapping,
                                                  full_machine_spec)
            .unwrap_milliseconds();

    MCMCOverMappedPCGConfig search_config =
        MCMCOverMappedPCGConfig{/*temperature=*/1.0,
                                /*num_iterations=*/100_n,
                                /*substitution_interval=*/5_n,
                                /*device_type=*/DeviceType::GPU};

    SearchResult result = mcmc_graph_optimize(
        pcg, cost_estimator, full_machine_spec, search_config);
    float runtime =
        task_simulator_estimate_forward_pass_time(result.pcg,
                                                  cost_estimator,
                                                  result.machine_mapping,
                                                  full_machine_spec)
            .unwrap_milliseconds();

    CHECK(runtime < base_runtime);
    CHECK(runtime < 100);
  }
}
