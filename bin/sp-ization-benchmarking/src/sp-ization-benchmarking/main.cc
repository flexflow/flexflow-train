/**
 * @file main.cc
 * @brief Benchmarking different SP-ization techniques on various graphs.
 *
 * @details
 * Algorithms:
 *  - work_duplicating_sp_ization_with_coalescing
 *  - naive_stratum_sync_sp_ization
 * Weight distributions:
 *  - Constant
 *  - Uniform(0, 1)
 *  - Binary(0, 100)
 *  - Chooser({1.0, 25.0, 500.0}) //sample uniformly from the given weights
 * Noise distributions:
 *  - NoNoise
 *  - GaussianNoise(1, 0.1)
 *  - UniformNoise(0.8, 1.25)
 * Graph types:
 *  ...
 *
 * @note To run the benchmark, go to build/normal/bin/sp_ization_benchmarking,
 * run make and then ./sp_ization_benchmarking
 */

#include "sp-ization-benchmarking/distributions.h"
#include "sp-ization-benchmarking/nasnet_bench_graph_generator.h"
#include "sp-ization-benchmarking/sample_graphs.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/graph/series_parallel/sp_ization/naive_stratum_sync.h"
#include "utils/graph/series_parallel/sp_ization/sp_ization_benchmark_result.dtg.h"
#include "utils/graph/series_parallel/sp_ization/sp_ization_combined_benchmark_result.dtg.h"
#include "utils/graph/series_parallel/sp_ization/work_duplicating_sp_ization.h"
#include <iomanip>
#include <iostream>
#include <string>

constexpr size_t REPEAT = 500;

using namespace FlexFlow;

template <typename D, typename N = NoNoise>
SpizationCombinedBenchmarkResult
    perform_benchmark_given_graph(DiGraphView const &g,
                                  D const &Dist,
                                  N const &Noise = NoNoise(),
                                  size_t repeat = REPEAT) {
  float work_dup_relative_work_increase = 0.0f;
  float work_dup_relative_critical_path_cost_increase = 0.0f;
  float work_dup_relative_num_dependencies_increase = 0.0f;

  float barrier_sync_relative_work_increase = 0.0f;
  float barrier_sync_relative_critical_path_cost_increase = 0.0f;
  float barrier_sync_relative_num_dependencies_increase = 0.0f;

  for (int i = 0; i < repeat; i++) {
    auto cost_map = make_cost_map(get_nodes(g), Dist);

    SeriesParallelDecomposition sp1 =
        work_duplicating_sp_ization_with_coalescing(g);
    SeriesParallelDecomposition sp2 = naive_stratum_sync_sp_ization(g);

    auto noisy_cost_map = add_noise_to_cost_map(cost_map, Noise);

    work_dup_relative_work_increase +=
        relative_work_increase(g, sp1, noisy_cost_map);
    work_dup_relative_critical_path_cost_increase +=
        relative_critical_path_cost_increase(g, sp1, noisy_cost_map);
    work_dup_relative_num_dependencies_increase +=
        relative_num_dependencies_increase(g, sp1);

    barrier_sync_relative_work_increase +=
        relative_work_increase(g, sp2, noisy_cost_map);
    barrier_sync_relative_critical_path_cost_increase +=
        relative_critical_path_cost_increase(g, sp2, noisy_cost_map);
    barrier_sync_relative_num_dependencies_increase +=
        relative_num_dependencies_increase(g, sp2);
  }

  SpizationBenchmarkResult work_duplicating{
      work_dup_relative_work_increase / repeat,
      work_dup_relative_critical_path_cost_increase / repeat,
      work_dup_relative_num_dependencies_increase / repeat,
  };

  SpizationBenchmarkResult barrier_sync{
      barrier_sync_relative_work_increase / repeat,
      barrier_sync_relative_critical_path_cost_increase / repeat,
      barrier_sync_relative_num_dependencies_increase / repeat,
  };

  return SpizationCombinedBenchmarkResult{{
      {"work_duplicating", work_duplicating},
      {"naive_stratum_sync", barrier_sync},
  }};
}

template <typename G, typename D, typename N = NoNoise>
SpizationCombinedBenchmarkResult
    perform_benchmark_given_graph_generator(G const &graph_generator,
                                            D const &Dist,
                                            N const &Noise = NoNoise(),
                                            size_t repeat = REPEAT) {
  float work_dup_relative_work_increase = 0.0f;
  float work_dup_relative_critical_path_cost_increase = 0.0f;
  float work_dup_relative_num_dependencies_increase = 0.0f;

  float barrier_sync_relative_work_increase = 0.0f;
  float barrier_sync_relative_critical_path_cost_increase = 0.0f;
  float barrier_sync_relative_num_dependencies_increase = 0.0f;

  for (int i = 0; i < repeat; i++) {
    DiGraphView g = graph_generator();
    auto cost_map = make_cost_map(get_nodes(g), Dist);

    SeriesParallelDecomposition sp1 =
        work_duplicating_sp_ization_with_coalescing(g);
    SeriesParallelDecomposition sp2 = naive_stratum_sync_sp_ization(g);

    auto noisy_cost_map = add_noise_to_cost_map(cost_map, Noise);

    work_dup_relative_work_increase +=
        relative_work_increase(g, sp1, noisy_cost_map);
    work_dup_relative_critical_path_cost_increase +=
        relative_critical_path_cost_increase(g, sp1, noisy_cost_map);
    work_dup_relative_num_dependencies_increase +=
        relative_num_dependencies_increase(g, sp1);

    barrier_sync_relative_work_increase +=
        relative_work_increase(g, sp2, noisy_cost_map);
    barrier_sync_relative_critical_path_cost_increase +=
        relative_critical_path_cost_increase(g, sp2, noisy_cost_map);
    barrier_sync_relative_num_dependencies_increase +=
        relative_num_dependencies_increase(g, sp2);
  }

  SpizationBenchmarkResult work_duplicating{
      work_dup_relative_work_increase / repeat,
      work_dup_relative_critical_path_cost_increase / repeat,
      work_dup_relative_num_dependencies_increase / repeat,
  };

  SpizationBenchmarkResult barrier_sync{
      barrier_sync_relative_work_increase / repeat,
      barrier_sync_relative_critical_path_cost_increase / repeat,
      barrier_sync_relative_num_dependencies_increase / repeat,
  };

  return SpizationCombinedBenchmarkResult{{
      {"work_duplicating", work_duplicating},
      {"naive_stratum_sync", barrier_sync},
  }};
}

void output_benchmark(SpizationCombinedBenchmarkResult const &combined_result,
                      std::string const &title) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Benchmark for " << title << std::endl;
  std::cout << "Technique | Work-Increase | Critical-Path-Increase | "
               "Dependencies-Increase"
            << std::endl;
  for (auto const &[technique, result] : combined_result.by_technique) {
    std::cout << technique << " | " << result.relative_work_increase << " | "
              << result.relative_critical_path_cost_increase << " | "
              << result.relative_num_dependencies_increase << std::endl;
  }
  std::cout << std::endl;
}

template <typename D, typename N = NoNoise>
void bench_mark_given_graph(std::string title,
                            DiGraphView const &g,
                            D const &Dist,
                            N const &Noise = NoNoise(),
                            size_t repeat = REPEAT) {
  output_benchmark(perform_benchmark_given_graph(g, Dist, Noise, repeat),
                   title);
}

template <typename G, typename D, typename N = NoNoise>
void bench_mark_given_graph_generator(std::string title,
                                      G const &generator,
                                      D const &Dist,
                                      N const &Noise = NoNoise(),
                                      size_t repeat = REPEAT) {
  output_benchmark(
      perform_benchmark_given_graph_generator(generator, Dist, Noise, repeat),
      title);
}

int main() {
  {
    DiGraph g = make_sample_dag_3();
    bench_mark_given_graph("sample_dag_3, Constant(1)", g, Constant(1));
    bench_mark_given_graph("sample_dag_3, Constant(1), UniformNoise(0.8, 1.25)",
                           g,
                           Constant(1),
                           UniformNoise(0.8, 1.25));
    bench_mark_given_graph("sample_dag_3, Constant(1), GaussianNoise(1, 0.1)",
                           g,
                           Constant(1),
                           GaussianNoise(1, 0.1));

    bench_mark_given_graph("sample_dag_3, Uniform(0,1)", g, Uniform(0, 1));
    bench_mark_given_graph(
        "sample_dag_3, Uniform(0,1), UniformNoise(0.8, 1.25)",
        g,
        Uniform(0, 1),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph("sample_dag_3, Uniform(0,1), GaussianNoise(1, 0.1)",
                           g,
                           Uniform(0, 1),
                           GaussianNoise(1, 0.1));

    bench_mark_given_graph("sample_dag_3, Binary(1, 80)", g, Binary(1, 80));
    bench_mark_given_graph(
        "sample_dag_3, Binary(1, 80), UniformNoise(0.8, 1.25)",
        g,
        Binary(1, 80),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph("sample_dag_3, Binary(1, 80), GaussianNoise(1, 0.1)",
                           g,
                           Binary(1, 80),
                           GaussianNoise(1, 0.1));

    bench_mark_given_graph("sample_dag_3, Chooser({1.0, 20.0, 500.0})",
                           g,
                           Chooser({1.0, 20.0, 500.0}));
    bench_mark_given_graph(
        "sample_dag_3, Chooser({1.0, 20.0, 500.0}), UniformNoise(0.8, 1.25)",
        g,
        Chooser({1.0, 20.0, 500.0}),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "sample_dag_3, Chooser({1.0, 20.0, 500.0}), GaussianNoise(1, 0.1)",
        g,
        Chooser({1.0, 20.0, 500.0}),
        GaussianNoise(1, 0.1));
  }

  {
    DiGraph g = make_taso_nasnet_cell();
    bench_mark_given_graph("taso_nasnet_cell, Constant(1)", g, Constant(1));
    bench_mark_given_graph(
        "taso_nasnet_cell, Constant(1), UniformNoise(0.8, 1.25)",
        g,
        Constant(1),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "taso_nasnet_cell, Constant(1), GaussianNoise(1, 0.1)",
        g,
        Constant(1),
        GaussianNoise(1, 0.1));

    bench_mark_given_graph("taso_nasnet_cell, Uniform(0,1)", g, Uniform(0, 1));
    bench_mark_given_graph(
        "taso_nasnet_cell, Uniform(0,1), UniformNoise(0.8, 1.25)",
        g,
        Uniform(0, 1),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "taso_nasnet_cell, Uniform(0,1), GaussianNoise(1, 0.1)",
        g,
        Uniform(0, 1),
        GaussianNoise(1, 0.1));

    bench_mark_given_graph("taso_nasnet_cell, Binary(1, 80)", g, Binary(1, 80));
    bench_mark_given_graph(
        "taso_nasnet_cell, Binary(1, 80), UniformNoise(0.8, 1.25)",
        g,
        Binary(1, 80),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "taso_nasnet_cell, Binary(1, 80), GaussianNoise(1, 0.1)",
        g,
        Binary(1, 80),
        GaussianNoise(1, 0.1));

    bench_mark_given_graph("taso_nasnet_cell, Chooser({1.0, 20.0, 500.0})",
                           g,
                           Chooser({1.0, 20.0, 500.0}));
    bench_mark_given_graph("taso_nasnet_cell, Chooser({1.0, 20.0, 500.0}), "
                           "UniformNoise(0.8, 1.25)",
                           g,
                           Chooser({1.0, 20.0, 500.0}),
                           UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "taso_nasnet_cell, Chooser({1.0, 20.0, 500.0}), GaussianNoise(1, 0.1)",
        g,
        Chooser({1.0, 20.0, 500.0}),
        GaussianNoise(1, 0.1));
  }

  {
    DiGraph g = make_parallel_chains(10, 5);
    bench_mark_given_graph("parallel_chains, Constant(1)", g, Constant(1));
    bench_mark_given_graph(
        "parallel_chains, Constant(1), UniformNoise(0.8, 1.25)",
        g,
        Constant(1),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "parallel_chains, Constant(1), GaussianNoise(1, 0.1)",
        g,
        Constant(1),
        GaussianNoise(1, 0.1));

    bench_mark_given_graph("parallel_chains, Uniform(0,1)", g, Uniform(0, 1));
    bench_mark_given_graph(
        "parallel_chains, Uniform(0,1), UniformNoise(0.8, 1.25)",
        g,
        Uniform(0, 1),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "parallel_chains, Uniform(0,1), GaussianNoise(1, 0.1)",
        g,
        Uniform(0, 1),
        GaussianNoise(1, 0.1));

    bench_mark_given_graph("parallel_chains, Binary(1, 80)", g, Binary(1, 80));
    bench_mark_given_graph(
        "parallel_chains, Binary(1, 80), UniformNoise(0.8, 1.25)",
        g,
        Binary(1, 80),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "parallel_chains, Binary(1, 80), GaussianNoise(1, 0.1)",
        g,
        Binary(1, 80),
        GaussianNoise(1, 0.1));

    bench_mark_given_graph("parallel_chains, Chooser({1.0, 20.0, 500.0})",
                           g,
                           Chooser({1.0, 20.0, 500.0}));
    bench_mark_given_graph(
        "parallel_chains, Chooser({1.0, 20.0, 500.0}), UniformNoise(0.8, 1.25)",
        g,
        Chooser({1.0, 20.0, 500.0}),
        UniformNoise(0.8, 1.25));
    bench_mark_given_graph(
        "parallel_chains, Chooser({1.0, 20.0, 500.0}), GaussianNoise(1, 0.1)",
        g,
        Chooser({1.0, 20.0, 500.0}),
        GaussianNoise(1, 0.1));
  }

  {

    auto generate_2_terminal_random_dag = []() {
      return make_2_terminal_random_dag(60, .12, 5);
    };
    size_t repeat = 100;
    bench_mark_given_graph_generator("make_2_terminal_random_dag, Constant(1)",
                                     generate_2_terminal_random_dag,
                                     Constant(1),
                                     NoNoise(),
                                     repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Constant(1), UniformNoise(0.8, 1.25)",
        generate_2_terminal_random_dag,
        Constant(1),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Constant(1), GaussianNoise(1, 0.1)",
        generate_2_terminal_random_dag,
        Constant(1),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph_generator("make_2_terminal_random_dag, Uniform(0,1)",
                                     generate_2_terminal_random_dag,
                                     Uniform(0, 1),
                                     NoNoise(),
                                     repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Uniform(0,1), UniformNoise(0.8, 1.25)",
        generate_2_terminal_random_dag,
        Uniform(0, 1),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Uniform(0,1), GaussianNoise(1, 0.1)",
        generate_2_terminal_random_dag,
        Uniform(0, 1),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Binary(1, 80)",
        generate_2_terminal_random_dag,
        Binary(1, 80),
        NoNoise(),
        repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Binary(1, 80), UniformNoise(0.8, 1.25)",
        generate_2_terminal_random_dag,
        Binary(1, 80),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Binary(1, 80), GaussianNoise(1, 0.1)",
        generate_2_terminal_random_dag,
        Binary(1, 80),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Chooser({1.0, 20.0, 500.0})",
        generate_2_terminal_random_dag,
        Chooser({1.0, 20.0, 500.0}),
        NoNoise(),
        repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Chooser({1.0, 20.0, 500.0}), "
        "UniformNoise(0.8, 1.25)",
        generate_2_terminal_random_dag,
        Chooser({1.0, 20.0, 500.0}),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "make_2_terminal_random_dag, Chooser({1.0, 20.0, 500.0}), "
        "GaussianNoise(1, 0.1)",
        generate_2_terminal_random_dag,
        Chooser({1.0, 20.0, 500.0}),
        GaussianNoise(1, 0.1),
        repeat);
  }

  {
    size_t repeat = 100;
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Constant(1)",
        generate_nasnet_bench_network,
        Constant(1),
        NoNoise(),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Constant(1), UniformNoise(0.8, 1.25)",
        generate_nasnet_bench_network,
        Constant(1),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Constant(1), GaussianNoise(1, 0.1)",
        generate_nasnet_bench_network,
        Constant(1),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Uniform(0,1)",
        generate_nasnet_bench_network,
        Uniform(0, 1),
        NoNoise(),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Uniform(0,1), UniformNoise(0.8, 1.25)",
        generate_nasnet_bench_network,
        Uniform(0, 1),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Uniform(0,1), GaussianNoise(1, 0.1)",
        generate_nasnet_bench_network,
        Uniform(0, 1),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Binary(1, 80)",
        generate_nasnet_bench_network,
        Binary(1, 80),
        NoNoise(),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Binary(1, 80), UniformNoise(0.8, 1.25)",
        generate_nasnet_bench_network,
        Binary(1, 80),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Binary(1, 80), GaussianNoise(1, 0.1)",
        generate_nasnet_bench_network,
        Binary(1, 80),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Chooser({1.0, 20.0, 500.0})",
        generate_nasnet_bench_network,
        Chooser({1.0, 20.0, 500.0}),
        NoNoise(),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Chooser({1.0, 20.0, 500.0}), "
        "UniformNoise(0.8, 1.25)",
        generate_nasnet_bench_network,
        Chooser({1.0, 20.0, 500.0}),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph_generator(
        "generate_nasnet_bench_network, Chooser({1.0, 20.0, 500.0}), "
        "GaussianNoise(1, 0.1)",
        generate_nasnet_bench_network,
        Chooser({1.0, 20.0, 500.0}),
        GaussianNoise(1, 0.1),
        repeat);
  }

  {
    size_t repeat = 10;
    DiGraph g = make_full_taso_nasnet(1, 1);
    bench_mark_given_graph("make_full_taso_nasnet, Constant(1)",
                           g,
                           Constant(1),
                           NoNoise(),
                           repeat);
    bench_mark_given_graph(
        "make_full_taso_nasnet, Constant(1), UniformNoise(0.8, 1.25)",
        g,
        Constant(1),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph(
        "make_full_taso_nasnet, Constant(1), GaussianNoise(1, 0.1)",
        g,
        Constant(1),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph("make_full_taso_nasnet, Uniform(0,1)",
                           g,
                           Uniform(0, 1),
                           NoNoise(),
                           repeat);
    bench_mark_given_graph(
        "make_full_taso_nasnet, Uniform(0,1), UniformNoise(0.8, 1.25)",
        g,
        Uniform(0, 1),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph(
        "make_full_taso_nasnet, Uniform(0,1), GaussianNoise(1, 0.1)",
        g,
        Uniform(0, 1),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph("make_full_taso_nasnet, Binary(1, 80)",
                           g,
                           Binary(1, 80),
                           NoNoise(),
                           repeat);
    bench_mark_given_graph(
        "make_full_taso_nasnet, Binary(1, 80), UniformNoise(0.8, 1.25)",
        g,
        Binary(1, 80),
        UniformNoise(0.8, 1.25),
        repeat);
    bench_mark_given_graph(
        "make_full_taso_nasnet, Binary(1, 80), GaussianNoise(1, 0.1)",
        g,
        Binary(1, 80),
        GaussianNoise(1, 0.1),
        repeat);

    bench_mark_given_graph("make_full_taso_nasnet, Chooser({1.0, 20.0, 500.0})",
                           g,
                           Chooser({1.0, 20.0, 500.0}),
                           NoNoise(),
                           repeat);
    bench_mark_given_graph("make_full_taso_nasnet, Chooser({1.0, 20.0, "
                           "500.0}), UniformNoise(0.8, 1.25)",
                           g,
                           Chooser({1.0, 20.0, 500.0}),
                           UniformNoise(0.8, 1.25),
                           repeat);
    bench_mark_given_graph("make_full_taso_nasnet, Chooser({1.0, 20.0, "
                           "500.0}), GaussianNoise(1, 0.1)",
                           g,
                           Chooser({1.0, 20.0, 500.0}),
                           GaussianNoise(1, 0.1),
                           repeat);
  }
}
