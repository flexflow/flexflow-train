#include "utils/orthotope/orthotope_bijective_projection.h"
#include "utils/containers/zip.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/unordered_set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("operator==(OrthotopeBijectiveProjection, OrthotopeBijectiveProjection)") {
    SUBCASE("if src num dims and dst num dims are the same, projections are equivalent") {
      orthotope_dim_idx_t src0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t src1 = orthotope_dim_idx_t{1};

      orthotope_dim_idx_t dst0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t dst1 = orthotope_dim_idx_t{1};

      OrthotopeBijectiveProjection p = make_orthotope_projection_from_map(
        {
          {src0, dst0},
          {src1, dst1},
        },
        /*reversed=*/false);

      CHECK(p == reverse_projection(p));
    }
  }

  TEST_CASE("get_all_bijective_projections_between") {
    SUBCASE("dst num dims greater than src num dims") {
      Orthotope src = Orthotope{{6, 4}};
      orthotope_dim_idx_t src0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t src1 = orthotope_dim_idx_t{1};

      Orthotope dst = Orthotope{{3, 4, 2}};
      orthotope_dim_idx_t dst0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t dst1 = orthotope_dim_idx_t{1};
      orthotope_dim_idx_t dst2 = orthotope_dim_idx_t{2};

      std::unordered_set<OrthotopeBijectiveProjection> result = get_all_bijective_projections_between(src, dst);
      std::unordered_set<OrthotopeBijectiveProjection> correct = {
        make_orthotope_projection_from_map({
          {dst0, src0},
          {dst1, src1},
          {dst2, src0},
        }, /*reversed=*/true),
      };

      CHECK(result == correct);
    }

    SUBCASE("src num dims greater than dst num dims") {
      Orthotope src = Orthotope{{3, 4, 2}};
      orthotope_dim_idx_t src0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t src1 = orthotope_dim_idx_t{1};
      orthotope_dim_idx_t src2 = orthotope_dim_idx_t{2};

      Orthotope dst = Orthotope{{6, 4}};
      orthotope_dim_idx_t dst0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t dst1 = orthotope_dim_idx_t{1};

      std::unordered_set<OrthotopeBijectiveProjection> result = get_all_bijective_projections_between(src, dst);
      std::unordered_set<OrthotopeBijectiveProjection> correct = {
        make_orthotope_projection_from_map({
          {src0, dst0},
          {src1, dst1},
          {src2, dst0},
        }, /*reversed=*/false),
      };

      CHECK(result == correct);
    }

    SUBCASE("multiple possible mappings") {
      Orthotope src = Orthotope{{3, 3}};
      orthotope_dim_idx_t src0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t src1 = orthotope_dim_idx_t{1};

      Orthotope dst = Orthotope{{3, 3}};
      orthotope_dim_idx_t dst0 = orthotope_dim_idx_t{0};
      orthotope_dim_idx_t dst1 = orthotope_dim_idx_t{1};

      std::unordered_set<OrthotopeBijectiveProjection> result = get_all_bijective_projections_between(src, dst);
      std::unordered_set<OrthotopeBijectiveProjection> correct = {
        make_orthotope_projection_from_map({
          {src0, dst0},
          {src1, dst1},
        }, /*reversed=*/false),
        make_orthotope_projection_from_map({
          {src0, dst1},
          {src1, dst0},
        }, /*reversed=*/false),
      };

      CHECK(result == correct);
    }

    SUBCASE("no possible mappings") {
      Orthotope src = Orthotope{{4, 3}};
      Orthotope dst = Orthotope{{6, 2}};

      std::unordered_set<OrthotopeBijectiveProjection> result = get_all_bijective_projections_between(src, dst);
      std::unordered_set<OrthotopeBijectiveProjection> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("project_into_1d") {
    SUBCASE("to 1d from 1d is identity") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{2}};
      Orthotope orthotope = Orthotope{{5}};

      int result = project_into_1d(orthotope, coord);
      int correct = 2;

      CHECK(result == correct);
    }

    SUBCASE("basic example") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{4, 1}};
      Orthotope orthotope = Orthotope{{5, 3}};

      int result = project_into_1d(orthotope, coord);
      int correct = 4 * 3 + 1;

      CHECK(result == correct);
    }

    SUBCASE("order matters") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{1, 4}};
      Orthotope orthotope = Orthotope{{3, 5}};

      int result = project_into_1d(orthotope, coord);
      int correct = 1 * 5 + 4;

      CHECK(result == correct);
    }

    SUBCASE("throws if coord is outside of orthotope") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{
        {2, 3, 1},
      };

      Orthotope orthotope = Orthotope{
        {5, 3, 2},
      };

      CHECK_THROWS(project_into_1d(orthotope, coord));
    }

    SUBCASE("throws if coord does not have same dimension as orthotope") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{
        {2, 3, 1},
      };

      Orthotope orthotope = Orthotope{
        {5, 3},
      };

      CHECK_THROWS(project_into_1d(orthotope, coord));
    }

    SUBCASE("returns 0 if orthotope is 0-dimensional") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{}};
      Orthotope orthotope = Orthotope{{}};

      int result = project_into_1d(orthotope, coord);
      int correct = 0;

      CHECK(result == correct);
    }
  }

  TEST_CASE("project_out_of_1d") {
    SUBCASE("from 1d to 1d is identity") {
      Orthotope orthotope = Orthotope{{5}};
      int coord = 2;

      OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
      OrthotopeCoordinate correct = OrthotopeCoordinate{{2}};

      CHECK(result == correct);
    }

    SUBCASE("basic example") {
      Orthotope orthotope = Orthotope{{5, 3}};
      int coord = 4 * 3 + 1;

      OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
      OrthotopeCoordinate correct = OrthotopeCoordinate{{4, 1}};

      CHECK(result == correct);
    }

    SUBCASE("orthotope dimension order matters") {
      Orthotope orthotope = Orthotope{{3, 5}};
      int coord = 1 * 5 + 4;

      OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
      OrthotopeCoordinate correct = OrthotopeCoordinate{{1, 4}};

      CHECK(result == correct);
    }

    SUBCASE("throws if coord would be projected outside of orthotope") {
      Orthotope orthotope = Orthotope{{5, 3}};

      SUBCASE("smallest coord outside of orthotope") {
        int coord = 15;

        CHECK_THROWS(project_out_of_1d(coord, orthotope));
      }

      SUBCASE("largest coord inside of orthotope") {
        int coord = 14;

        OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
        OrthotopeCoordinate correct = OrthotopeCoordinate{{4, 2}};

        CHECK(result == correct);
      }
    }

    SUBCASE("if dst orthotope is 0-dimensional") {
      Orthotope orthotope = Orthotope{{}};

      SUBCASE("returns 0-d coord if input coord is 0") {
        int input_coord = 0;

        OrthotopeCoordinate result = project_out_of_1d(input_coord, orthotope);
        OrthotopeCoordinate correct = OrthotopeCoordinate{{}};

        CHECK(result == correct);
      }

      SUBCASE("throws if input coord is anything other than zero") {
        int input_coord = 1;

        CHECK_THROWS(project_out_of_1d(input_coord, orthotope));
      }
    }
  }

  TEST_CASE("project_coordinate_through") {
    Orthotope src = Orthotope{
      {2, 3},
    };

    Orthotope dst = Orthotope{
      {6},
    };

    OrthotopeBijectiveProjection proj = OrthotopeBijectiveProjection{
      {orthotope_dim_idx_t{0}, orthotope_dim_idx_t{0}},
      /*reversed=*/false,
    };

    OrthotopeCoordinate src_coord = OrthotopeCoordinate{
      {1, 2},
    };
    OrthotopeCoordinate dst_coord = OrthotopeCoordinate{
      {1*3+2},
    };

    SUBCASE("forward") {
      OrthotopeCoordinate result = project_coordinate_through(proj, src, src_coord, dst);
      OrthotopeCoordinate correct = dst_coord;

      CHECK(result == correct);
    }

    SUBCASE("backward") {
      OrthotopeBijectiveProjection reversed = reverse_projection(proj);

      OrthotopeCoordinate result = project_coordinate_through(reversed, dst, dst_coord, src);
      OrthotopeCoordinate correct = src_coord;

      CHECK(result == correct);
    }
  }
}
