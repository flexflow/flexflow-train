#include "compiler/mcmc/mcmc_over_mapped_pcg.h"
#include "compiler/search_result.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "internal/runtime_only_cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/replica_type.dtg.h"
#include "op-attrs/shard_parallel_dim.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mcmc_over_mapped_pcg") {
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

    MachineSpaceCoordinate default_device = MachineSpaceCoordinate{
        /*node_idx=*/0_n,
        /*device_idx=*/0_n,
    };

    SearchResult lifted = trivial_search_result_for_cg(cg, default_device);
    ParallelComputationGraph initial_pcg = lifted.pcg;
    MachineMapping initial_mapping = lifted.machine_mapping;

    RuntimeOnlyCostEstimator cost_estimator =
        make_fake_constant_runtime_only_cost_estimator(
            /*forward_op_cost=*/10_ms,
            /*backward_op_cost=*/10_ms,
            /*comm_cost=*/1_ms);

    MachineSpecification full_machine_spec = MachineSpecification{
        MachineComputeSpecification{
            /*num_nodes=*/3_p,
            /*num_cpus_per_node=*/3_p,
            /*num_gpus_per_node=*/3_p,
        },
        MachineInterconnectSpecification{
            /*inter_node_bandwidth=*/bytes_per_second_t{1.0f},
            /*intra_node_bandwidth=*/bytes_per_second_t{1.0f},
        },
    };

    std::vector<Substitution> substitution_set =
        get_substitution_set(full_machine_spec.compute_specification);

    auto search_with_config =
        [&](MCMCOverMappedPCGConfig const &search_config) -> float {
      SearchResult search_result = mcmc_over_mapped_pcg(initial_pcg,
                                                        cost_estimator,
                                                        full_machine_spec,
                                                        search_config,
                                                        substitution_set,
                                                        initial_mapping);
      float result_fwd_pass_time = task_simulator_estimate_forward_pass_time(
                                       search_result.pcg,
                                       cost_estimator,
                                       search_result.machine_mapping,
                                       full_machine_spec)
                                       .unwrap_milliseconds();

      return result_fwd_pass_time;
    };

    MCMCOverMappedPCGConfig no_search = MCMCOverMappedPCGConfig{
        /*temperature=*/1.0,
        /*num_iterations=*/1_n,
        /*substitution_frequency=*/0.2,
    };

    MCMCOverMappedPCGConfig search_config = MCMCOverMappedPCGConfig{
        /*temperature=*/1.0,
        /*num_iterations=*/100_n,
        /*substitution_frequency=*/0.2,
    };

    float base_fwd_pass_time = search_with_config(no_search);
    float optimized_fwd_pass_time = search_with_config(search_config);

    CHECK(optimized_fwd_pass_time < base_fwd_pass_time * 0.8);
  }
}
