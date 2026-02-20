#include "utils/orthotope/orthotope_coord.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("restrict_orthotope_coord_to_dims") {
    OrthotopeCoord coord = OrthotopeCoord{{
        3_n,
        5_n,
    }};

    SUBCASE("allowed_dims is a superset of the coord's dims") {
      std::set<nonnegative_int> allowed_dims = {0_n, 1_n, 2_n, 3_n};

      OrthotopeCoord result =
          restrict_orthotope_coord_to_dims(coord, allowed_dims);

      OrthotopeCoord correct = coord;

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims is the same as the coord's dims") {
      std::set<nonnegative_int> allowed_dims = {0_n, 1_n};

      OrthotopeCoord result =
          restrict_orthotope_coord_to_dims(coord, allowed_dims);

      OrthotopeCoord correct = coord;

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims overlaps the coord's dims") {
      std::set<nonnegative_int> allowed_dims = {1_n, 3_n};

      OrthotopeCoord result =
          restrict_orthotope_coord_to_dims(coord, allowed_dims);

      OrthotopeCoord correct = OrthotopeCoord{{
          5_n,
      }};

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims is a subset of the coord's dims") {
      std::set<nonnegative_int> allowed_dims = {0_n};

      OrthotopeCoord result =
          restrict_orthotope_coord_to_dims(coord, allowed_dims);

      OrthotopeCoord correct = OrthotopeCoord{{
          3_n,
      }};

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims is empty") {
      std::set<nonnegative_int> allowed_dims = {};

      OrthotopeCoord result =
          restrict_orthotope_coord_to_dims(coord, allowed_dims);

      OrthotopeCoord correct = OrthotopeCoord{{}};

      CHECK(result == correct);
    }

    SUBCASE("coord is empty") {
      std::set<nonnegative_int> allowed_dims = {0_n, 1_n};

      OrthotopeCoord result =
          restrict_orthotope_coord_to_dims(OrthotopeCoord{{}}, allowed_dims);

      OrthotopeCoord correct = OrthotopeCoord{{}};

      CHECK(result == correct);
    }
  }
}
