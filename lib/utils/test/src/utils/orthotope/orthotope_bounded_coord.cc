#include "utils/orthotope/orthotope_bounded_coord.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/rapidcheck/doctest.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("OrthotopeBoundedCoord constructor") {
    Orthotope bounds = Orthotope{
        {4_p, 3_p, 5_p},
    };

    SUBCASE("coord dimensionality does not match bounds dimensionality") {
      OrthotopeCoord coord = OrthotopeCoord{
          {2_n, 1_n},
      };

      CHECK_THROWS(OrthotopeBoundedCoord{coord, bounds});
    }

    SUBCASE("coord is outside of bounds") {
      OrthotopeCoord coord = OrthotopeCoord{
          {1_n, 3_n, 1_n},
      };

      CHECK_THROWS(OrthotopeBoundedCoord{coord, bounds});
    }

    SUBCASE("coord is inside bounds") {
      OrthotopeCoord coord = OrthotopeCoord{
          {1_n, 2_n, 1_n},
      };

      // check does not throw
      OrthotopeBoundedCoord{coord, bounds};
    }
  }

  TEST_CASE("orthotope_bounded_coord_num_dims") {
    OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
            {2_n, 1_n, 3_n},
        },
        /*bounds=*/
        Orthotope{
            {4_p, 3_p, 5_p},
        },
    };

    nonnegative_int result = orthotope_bounded_coord_num_dims(input);
    nonnegative_int correct = 3_n;

    CHECK(result == correct);
  }

  TEST_CASE("orthotope_bounded_coord_product") {
    std::vector<OrthotopeBoundedCoord> inputs = {
        OrthotopeBoundedCoord{
            /*coord=*/OrthotopeCoord{
                {2_n, 1_n, 3_n},
            },
            /*bounds=*/
            Orthotope{
                {4_p, 3_p, 5_p},
            },
        },
        OrthotopeBoundedCoord{
            /*coord=*/OrthotopeCoord{
                {3_n, 1_n},
            },
            /*bounds=*/
            Orthotope{
                {6_p, 4_p},
            },
        },
    };

    OrthotopeBoundedCoord result = orthotope_bounded_coord_product(inputs);
    OrthotopeBoundedCoord correct = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
            {2_n, 1_n, 3_n, 3_n, 1_n},
        },
        /*bounds=*/
        Orthotope{
            {4_p, 3_p, 5_p, 6_p, 4_p},
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("orthotope_bounded_coord_slice_first") {
    SUBCASE("input num dims is 0") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{}},
          /*bounds=*/Orthotope{{}},
      };

      CHECK_THROWS(orthotope_bounded_coord_slice_first(input));
    }

    SUBCASE("input num dims is 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{
              {2_n},
          },
          /*bounds=*/
          Orthotope{
              {4_p},
          },
      };

      std::pair<BoundedComponent, OrthotopeBoundedCoord> result =
          orthotope_bounded_coord_slice_first(input);
      std::pair<BoundedComponent, OrthotopeBoundedCoord> correct = {
          BoundedComponent{2_n, 4_p},
          OrthotopeBoundedCoord{
              /*coord=*/OrthotopeCoord{{}},
              /*bounds=*/Orthotope{{}},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("input num dims is > 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{
              {2_n, 1_n, 3_n},
          },
          /*bounds=*/
          Orthotope{
              {4_p, 3_p, 5_p},
          },
      };

      std::pair<BoundedComponent, OrthotopeBoundedCoord> result =
          orthotope_bounded_coord_slice_first(input);
      std::pair<BoundedComponent, OrthotopeBoundedCoord> correct = {
          BoundedComponent{2_n, 4_p},
          OrthotopeBoundedCoord{
              /*coord=*/OrthotopeCoord{{1_n, 3_n}},
              /*bounds=*/Orthotope{{3_p, 5_p}},
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("orthotope_bounded_coord_slice_last") {
    SUBCASE("input num dims is 0") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{}},
          /*bounds=*/Orthotope{{}},
      };

      CHECK_THROWS(orthotope_bounded_coord_slice_last(input));
    }

    SUBCASE("input num dims is 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{
              {2_n},
          },
          /*bounds=*/
          Orthotope{
              {4_p},
          },
      };

      std::pair<OrthotopeBoundedCoord, BoundedComponent> result =
          orthotope_bounded_coord_slice_last(input);
      std::pair<OrthotopeBoundedCoord, BoundedComponent> correct = {
          OrthotopeBoundedCoord{
              /*coord=*/OrthotopeCoord{{}},
              /*bounds=*/Orthotope{{}},
          },
          BoundedComponent{2_n, 4_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("input num dims is > 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{
              {2_n, 1_n, 3_n},
          },
          /*bounds=*/
          Orthotope{
              {4_p, 3_p, 5_p},
          },
      };

      std::pair<OrthotopeBoundedCoord, BoundedComponent> result =
          orthotope_bounded_coord_slice_last(input);
      std::pair<OrthotopeBoundedCoord, BoundedComponent> correct = {
          OrthotopeBoundedCoord{
              /*coord=*/OrthotopeCoord{{2_n, 1_n}},
              /*bounds=*/Orthotope{{4_p, 3_p}},
          },
          BoundedComponent{3_n, 5_p},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("flatten_orthotope_bounded_coord") {
    SUBCASE("input num dims is 0") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{}},
          /*bounds=*/Orthotope{{}},
      };

      std::optional<BoundedComponent> result =
          flatten_orthotope_bounded_coord(input);
      std::optional<BoundedComponent> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input num dims is 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{
              {2_n},
          },
          /*bounds=*/
          Orthotope{
              {4_p},
          },
      };

      std::optional<BoundedComponent> result =
          flatten_orthotope_bounded_coord(input);
      std::optional<BoundedComponent> correct = BoundedComponent{
          /*component=*/2_n,
          /*bound=*/4_p,
      };

      CHECK(result == correct);
    }

    SUBCASE("input num dims is > 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{
              {2_n, 1_n, 3_n},
          },
          /*bounds=*/
          Orthotope{
              {4_p, 3_p, 5_p},
          },
      };

      std::optional<BoundedComponent> result =
          flatten_orthotope_bounded_coord(input);
      std::optional<BoundedComponent> correct = BoundedComponent{
          /*component=*/2_n * 3_p * 5_p + 1_n * 5_p + 3_n,
          /*bound=*/4_p * 3_p * 5_p,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("OrthotopeBoundedCoord json serialization and deserialization") {
    RC_SUBCASE([](OrthotopeBoundedCoord const &original) -> bool {
      nlohmann::json serialized = original;
      OrthotopeBoundedCoord deserialized =
          serialized.get<OrthotopeBoundedCoord>();

      return deserialized == original;
    });
  }

  TEST_CASE("Arbitrary<OrthotopeBoundedCoord>") {
    RC_SUBCASE([](OrthotopeBoundedCoord const &) {});
  }
}
