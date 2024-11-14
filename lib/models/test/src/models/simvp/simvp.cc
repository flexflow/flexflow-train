#include "models/simvp/simvp.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_simvp_computation_graph") {
    SimVPConfig config = get_default_simvp_config();

    ComputationGraph result = get_simvp_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 1;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}
