#include "compiler/mcmc/generic_mcmc_algorithm.h"
#include "doctest/doctest.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("generic_mcmc_algorithm") {
    float starting_state = 0.1;
    auto generating_func = [](float x,
                              nonnegative_int i) -> std::optional<float> {
      float new_x = x + (randf() - 0.5) / (i.unwrap_nonnegative() + 1);
      if (new_x < 0) {
        return std::nullopt;
      }
      if (new_x > 1) {
        return std::nullopt;
      }
      return new_x;
    };
    auto scoring_func = [](float x) { return (x - 0.5) * (x - 0.5); };
    GenericMCMCConfig config = GenericMCMCConfig{/*temperature*/ 1.0,
                                                 /*num_iterations*/ 10_n};
    Generic_MCMC_state<float, float> result =
        minimize_score(starting_state, generating_func, scoring_func, config);
    float answer = result.get_state();
    float error = result.get_score();
    CHECK(answer > 0.49);
    CHECK(answer < 0.51);
    CHECK(error >= 0);
    CHECK(error < 0.01);
  }
}
