#include "models/yolov10/yolov10.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_yolov10_computation_graph") {
    YOLOv10Config config = get_default_yolov10_config();

    ComputationGraph result = get_yolov10_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 327;
      CHECK(result_num_layers == correct_num_layers);
    }
  }
}
