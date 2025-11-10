#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H

#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_projection.dtg.h"
#include "utils/orthotope/down_projection.h"
#include "utils/orthotope/eq_projection.h"
#include "utils/orthotope/up_projection.h"
#include "utils/overload.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"

namespace FlexFlow {

template <typename L, typename R>
DimProjection<L, R> dim_projection_identity_map(DimDomain<L> const &input_domain,
                                                DimDomain<R> const &output_domain,
                                                DimOrdering<L> const &input_dim_ordering,
                                                DimOrdering<R> const &output_dim_ordering) {
  
  std::vector<L> input_dims = sorted_by(get_domain_dims(input_domain),
                                        input_dim_ordering.lt);

  std::vector<R> output_dims = sorted_by(get_domain_dims(output_domain),
                                         output_dim_ordering.lt);

  return DimProjection{
    EqProjection{
      bidict_from_keys_and_values(input_dims, output_dims),
    },
  };
}

template <typename L, typename R>
std::unordered_set<L>
    input_dims_of_projection(DimProjection<L, R> const &projection) {
  return projection.template visit<std::unordered_set<L>>(overload{
      [](UpProjection<L, R> const &p) {
        return input_dims_of_up_projection(p);
      },
      [](EqProjection<L, R> const &p) {
        return input_dims_of_eq_projection(p);
      },
      [](DownProjection<L, R> const &p) {
        return input_dims_of_down_projection(p);
      },
  });
}

template <typename L, typename R>
std::unordered_set<R>
    output_dims_of_projection(DimProjection<L, R> const &projection) {
  return projection.template visit<std::unordered_set<R>>(overload{
      [](UpProjection<L, R> const &p) {
        return output_dims_of_up_projection(p);
      },
      [](EqProjection<L, R> const &p) {
        return output_dims_of_eq_projection(p);
      },
      [](DownProjection<L, R> const &p) {
        return output_dims_of_down_projection(p);
      },
  });
};

template <typename L, typename R>
DimProjection<R, L> invert_dim_projection(DimProjection<L, R> const &projection) {
  return projection.template visit<DimProjection<R, L>>(overload{
      [](UpProjection<L, R> const &p) {
        return DimProjection<R, L>{
          invert_up_projection(p),
        };
      },
      [](EqProjection<L, R> const &p) {
        return DimProjection<R, L>{
          invert_eq_projection(p),
        };
      },
      [](DownProjection<L, R> const &p) {
        return DimProjection<R, L>{
          invert_down_projection(p),
        };
      },
  });
}

template <typename L, typename R>
DimCoord<R> compute_dim_projection(DimProjection<L, R> const &projection,
                                   DimCoord<L> const &input_coord,
                                   DimDomain<L> const &input_domain,
                                   DimDomain<R> const &output_domain,
                                   DimOrdering<L> const &input_dim_ordering,
                                   DimOrdering<R> const &output_dim_ordering) {
  DimCoord<L> lifted_input_coord = lift_dim_coord(input_coord, get_domain_dims(input_domain));

  ASSERT(dim_domain_contains_coord(input_domain, input_coord),
         input_domain,
         input_coord);

  {
    std::unordered_set<L> nontrivial_input_domain_dims = get_nontrivial_domain_dims(input_domain);
    std::unordered_set<L> projection_input_dims = input_dims_of_projection(projection);
    std::unordered_set<L> all_input_domain_dims = get_domain_dims(input_domain);

    ASSERT(is_subseteq_of(nontrivial_input_domain_dims, projection_input_dims),
           nontrivial_input_domain_dims,
           projection_input_dims);
    ASSERT(is_subseteq_of(projection_input_dims, all_input_domain_dims),
           projection_input_dims,
           all_input_domain_dims);
  }

  {
    std::unordered_set<R> nontrivial_output_domain_dims = get_nontrivial_domain_dims(output_domain);
    std::unordered_set<R> projection_output_dims = output_dims_of_projection(projection);
    std::unordered_set<R> all_output_domain_dims = get_domain_dims(output_domain);

    ASSERT(is_subseteq_of(nontrivial_output_domain_dims, projection_output_dims),
           nontrivial_output_domain_dims,
           projection_output_dims);
    ASSERT(is_subseteq_of(projection_output_dims, all_output_domain_dims),
           projection_output_dims,
           all_output_domain_dims);
  }

  DimCoord<R> output_coord = projection.template visit<DimCoord<R>>(overload{
      [&](UpProjection<L, R> const &p) -> DimCoord<R> {
        return compute_up_projection(
            p, lifted_input_coord, output_domain, output_dim_ordering);
      },
      [&](EqProjection<L, R> const &p) -> DimCoord<R> {
        return compute_eq_projection(p, lifted_input_coord);
      },
      [&](DownProjection<L, R> const &p) -> DimCoord<R> {
        return compute_down_projection(
            p, lifted_input_coord, input_domain, input_dim_ordering);
      },
  });

  DimCoord<R> lifted_output_coord = lift_dim_coord(output_coord, get_domain_dims(output_domain));

  ASSERT(dim_domain_contains_coord(output_domain, lifted_output_coord),
         output_domain,
         lifted_output_coord,
         input_domain,
         lifted_input_coord);

  return lifted_output_coord;
}


template <typename T1, typename T2, typename T3>
DimProjection<T1, T3> right_compose_eq_projection(
   DimProjection<T1, T2> const &lhs,
   EqProjection<T2, T3> const &rhs) {
  return lhs.template visit<DimProjection<T1, T3>>(overload{
    [&](UpProjection<T1, T2> const &lhs_up_proj) {
      return DimProjection<T1, T3>{
        compose_up_projections(
          lhs_up_proj,
          up_from_eq_proj(rhs)),
      };
    },
    [&](EqProjection<T1, T2> const &lhs_eq_proj) {
      return DimProjection<T1, T3>{
        compose_eq_projections(
          lhs_eq_proj,
          rhs),
      };
    },
    [&](DownProjection<T1, T2> const &lhs_down_proj) {
      return DimProjection<T1, T3>{
        compose_down_projections(
          lhs_down_proj,
          down_from_eq_proj(rhs)),
      };
    },
  });
}

template <typename T1, typename T2, typename T3>
DimProjection<T1, T3> left_compose_eq_projection(
    EqProjection<T1, T2> const &lhs,
    DimProjection<T2, T3> const &rhs) {
  return rhs.template visit<DimProjection<T1, T3>>(overload{
    [&](UpProjection<T2, T3> const &rhs_up_proj) {
      return DimProjection<T1, T3>{
        compose_up_projections(
          up_from_eq_proj(lhs), 
          rhs_up_proj),
      };
    },
    [&](EqProjection<T2, T3> const &rhs_eq_proj) {
      return DimProjection<T1, T3>{
        compose_eq_projections(
          lhs,
          rhs_eq_proj),
      };
    },
    [&](DownProjection<T2, T3> const &rhs_down_proj) {
      return DimProjection{
        compose_down_projections(
          down_from_eq_proj(lhs),
          rhs_down_proj),
      };
    },
  });
}

template <typename T1, typename T2, typename T3>
DimProjection<T1, T3> compose_dim_projections(
  DimProjection<T1, T2> const &lhs,
  DimProjection<T2, T3> const &rhs) {

  if (lhs.is_eq_proj()) {
    return DimProjection{
      left_compose_eq_projection(
        lhs.require_eq_proj(),
        rhs),
    };
  } else if (rhs.is_eq_proj()) {
    return DimProjection{
      right_compose_eq_projection(
        lhs,
        rhs.require_eq_proj()),
    }; 
  } else if (lhs.is_up_proj() && rhs.is_up_proj()) {
    return DimProjection{
      compose_up_projections(
        lhs.require_up_proj(),
        rhs.require_up_proj()),
    }; 
  } else if (lhs.is_down_proj() && rhs.is_down_proj()) {
    return DimProjection{
      compose_down_projections(
        lhs.require_down_proj(),
        rhs.require_down_proj()),
    };    
  } else {
    PANIC("Cannot compose projections", lhs, rhs);
  }
} 

} // namespace FlexFlow

#endif
