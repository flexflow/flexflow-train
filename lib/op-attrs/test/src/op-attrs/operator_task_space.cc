#include "op-attrs/operator_task_space.h"
#include "utils/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_task_space_coordinates") {

    SUBCASE("OperatorTaskSpace has 0 dimensions") {
      OperatorTaskSpace task = OperatorTaskSpace{MinimalOrthotope{{}}};

      std::unordered_set<TaskSpaceCoordinate> correct = {
          TaskSpaceCoordinate{OrthotopeCoord{{}}}};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }

    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{2_ge2, 2_ge2}}};

      std::unordered_set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n}}},
      }};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }

    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{3_ge2, 2_ge2, 2_ge2}}};

      std::unordered_set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n, 1_n}}},
      }};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }
  }

  TEST_CASE("get_task_space_maximum_coordinate") {
    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{3_ge2, 2_ge2}}};

      TaskSpaceCoordinate correct =
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n}}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }

    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{3_ge2, 2_ge2, 4_ge2}}};

      TaskSpaceCoordinate correct =
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n, 3_n}}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }
  }
}
