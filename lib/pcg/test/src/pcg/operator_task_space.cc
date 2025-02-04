#include "pcg/operator_task_space.h"
#include "utils/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_task_space_coordinates") {

    SUBCASE("OperatorTaskSpace has 0 dimensions") {
      OperatorTaskSpace task = OperatorTaskSpace{{}};

      std::unordered_set<TaskSpaceCoordinate> correct = {
          TaskSpaceCoordinate{{}}};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }
    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{{2_n, 2_n}};

      std::unordered_set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{{0_n, 0_n}},
          TaskSpaceCoordinate{{0_n, 1_n}},
          TaskSpaceCoordinate{{1_n, 0_n}},
          TaskSpaceCoordinate{{1_n, 1_n}},
      }};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }
    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{{1_n, 2_n, 2_n}};

      std::unordered_set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{{0_n, 0_n, 0_n}},
          TaskSpaceCoordinate{{0_n, 0_n, 1_n}},
          TaskSpaceCoordinate{{0_n, 1_n, 0_n}},
          TaskSpaceCoordinate{{0_n, 1_n, 1_n}},
      }};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }
  }
  TEST_CASE("get_task_space_maximum_coordinate") {
    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{{3_n, 2_n}};

      TaskSpaceCoordinate correct = TaskSpaceCoordinate{{2_n, 1_n}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }
    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{{3_n, 2_n, 4_n}};

      TaskSpaceCoordinate correct = TaskSpaceCoordinate{{2_n, 1_n, 3_n}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }
  }
}
