#include "compiler/mcmc/generic_mcmc_algorithm.h"
#include "doctest/doctest.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("generic_mcmc_algorithm") {
    float starting_state = 0.1;
    auto sampler = [](float x) -> std::optional<float> {
      float new_x = x + (randf() - 0.5);
      if (new_x < 0) {
        return std::nullopt;
      }
      if (new_x > 1) {
        return std::nullopt;
      }
      return new_x;
    };
    auto cost = [](float x) { return (x - 0.5) * (x - 0.5); };
    GenericMCMCConfig config = GenericMCMCConfig{/*temperature=*/1.0,
                                                 /*num_iterations=*/100_n};
    float answer = run_mcmc(starting_state, sampler, cost, config);
    float error = cost(answer);
    CHECK(answer > 0.47);
    CHECK(answer < 0.53);
    CHECK(error >= 0);
    CHECK(error < 0.001);
  }
}
