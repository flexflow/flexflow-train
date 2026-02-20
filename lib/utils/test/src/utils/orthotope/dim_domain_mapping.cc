#include "utils/orthotope/dim_domain_mapping.h"
#include "utils/orthotope/dim_ordering.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dim_domain_mapping_from_projection") {
    DimDomain<int> input_domain = DimDomain<int>{{
        {7, 6_p},
        {1, 2_p},
    }};

    std::string dim_a = "a";
    std::string dim_b = "b";
    std::string dim_c = "c";
    std::string dim_d = "d";

    DimDomain<std::string> output_domain = DimDomain<std::string>{{
        {dim_a, 2_p},
        {dim_b, 2_p},
        {dim_c, 3_p},
        {dim_d, 1_p},
    }};

    DimOrdering<int> input_dim_ordering = make_dim_ordering_from_vector<int>({
        1,
        7,
    });

    DimOrdering<std::string> output_dim_ordering =
        make_dim_ordering_from_vector<std::string>({
            dim_a,
            dim_b,
            dim_c,
            dim_d,
        });

    DimProjection<int, std::string> projection = DimProjection{
        UpProjection{
            OneToMany<int, std::string>{
                {7, {"a", "c", "d"}},
                {1, {"b"}},
            },
        },
    };

    DimDomainMapping<int, std::string> result =
        dim_domain_mapping_from_projection(
            /*projection=*/projection,
            /*l_domain=*/input_domain,
            /*r_domain=*/output_domain,
            /*l_dim_ordering=*/input_dim_ordering,
            /*r_dim_ordering=*/output_dim_ordering);

    auto mk_input_coord = [](int dim7, int dim1) {
      return DimCoord<int>{{
          {7, nonnegative_int{dim7}},
          {1, nonnegative_int{dim1}},
      }};
    };

    auto mk_output_coord = [&](int a, int b, int c, int d) {
      return DimCoord<std::string>{{
          {dim_a, nonnegative_int{a}},
          {dim_b, nonnegative_int{b}},
          {dim_c, nonnegative_int{c}},
          {dim_d, nonnegative_int{d}},
      }};
    };

    DimDomainMapping<int, std::string> correct =
        DimDomainMapping<int, std::string>{
            /*coord_mapping=*/bidict<DimCoord<int>, DimCoord<std::string>>{
                {mk_input_coord(0, 0), mk_output_coord(0, 0, 0, 0)},
                {mk_input_coord(1, 0), mk_output_coord(0, 0, 1, 0)},
                {mk_input_coord(2, 0), mk_output_coord(0, 0, 2, 0)},
                {mk_input_coord(3, 0), mk_output_coord(1, 0, 0, 0)},
                {mk_input_coord(4, 0), mk_output_coord(1, 0, 1, 0)},
                {mk_input_coord(5, 0), mk_output_coord(1, 0, 2, 0)},
                {mk_input_coord(0, 1), mk_output_coord(0, 1, 0, 0)},
                {mk_input_coord(1, 1), mk_output_coord(0, 1, 1, 0)},
                {mk_input_coord(2, 1), mk_output_coord(0, 1, 2, 0)},
                {mk_input_coord(3, 1), mk_output_coord(1, 1, 0, 0)},
                {mk_input_coord(4, 1), mk_output_coord(1, 1, 1, 0)},
                {mk_input_coord(5, 1), mk_output_coord(1, 1, 2, 0)},
            },
            /*l_domain=*/input_domain,
            /*r_domain=*/output_domain,
        };

    CHECK(result == correct);
  }

  TEST_CASE("empty_dim_domain_mapping") {
    empty_dim_domain_mapping<int, std::string>();
  }
}
