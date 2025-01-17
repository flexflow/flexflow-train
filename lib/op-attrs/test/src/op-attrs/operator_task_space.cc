#include "op-attrs/operator_task_space.h"
#include "utils/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_task_space_coordinates") {

    SUBCASE("OperatorTaskSpace has 0 dimensions") {
      OperatorTaskSpace task = OperatorTaskSpace{Orthotope{{}}};

      std::unordered_set<TaskSpaceCoordinate> correct = {
          TaskSpaceCoordinate{OrthotopeCoordinate{{}}}};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }
    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{Orthotope{{2, 2}}};

      std::unordered_set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{OrthotopeCoordinate{{0, 0}}},
          TaskSpaceCoordinate{OrthotopeCoordinate{{0, 1}}},
          TaskSpaceCoordinate{OrthotopeCoordinate{{1, 0}}},
          TaskSpaceCoordinate{OrthotopeCoordinate{{1, 1}}},
      }};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }
    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{Orthotope{{1, 2, 2}}};

      std::unordered_set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{OrthotopeCoordinate{{0, 0, 0}}},
          TaskSpaceCoordinate{OrthotopeCoordinate{{0, 0, 1}}},
          TaskSpaceCoordinate{OrthotopeCoordinate{{0, 1, 0}}},
          TaskSpaceCoordinate{OrthotopeCoordinate{{0, 1, 1}}},
      }};
      std::unordered_set<TaskSpaceCoordinate> result =
          get_task_space_coordinates(task);
      CHECK(correct == result);
    }
  }
  TEST_CASE("get_task_space_maximum_coordinate") {
    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{Orthotope{{3, 2}}};

      TaskSpaceCoordinate correct = TaskSpaceCoordinate{OrthotopeCoordinate{{2, 1}}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }
    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task = OperatorTaskSpace{Orthotope{{3, 2, 4}}};

      TaskSpaceCoordinate correct = TaskSpaceCoordinate{OrthotopeCoordinate{{2, 1, 3}}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }
  }
}
