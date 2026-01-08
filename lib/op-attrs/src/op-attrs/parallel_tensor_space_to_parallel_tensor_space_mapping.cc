#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"

namespace FlexFlow {

ParallelTensorSpaceToParallelTensorSpaceMapping
    parallel_tensor_space_mapping_from_projection(
        DimProjection<parallel_tensor_dim_idx_t,
                      parallel_tensor_dim_idx_t> const &projection,
        ParallelTensorDimDegrees const &l_degrees,
        ParallelTensorDimDegrees const &r_degrees) {

  // TODO(@lockshaw)(#pr):
  // {
  //   std::unordered_set<parallel_tensor_dim_idx_t>
  //     l_dims =
  //     unordered_set_of(get_nontrivial_parallel_tensor_dim_indices(l_degrees));
  //   std::unordered_set<parallel_tensor_dim_idx_t>
  //     projection_input_dims = input_dims_of_projection(projection);
  //
  //   ASSERT(l_dims == projection_input_dims);
  // }
  //
  // {
  //   std::unordered_set<parallel_tensor_dim_idx_t>
  //     r_dims =
  //     unordered_set_of(get_nontrivial_parallel_tensor_dim_indices(r_degrees));
  //   std::unordered_set<parallel_tensor_dim_idx_t>
  //     projection_output_dims = output_dims_of_projection(projection);
  //
  //   ASSERT(r_dims == projection_output_dims);
  // }

  return ParallelTensorSpaceToParallelTensorSpaceMapping{
      dim_domain_mapping_from_projection(
          /*projection=*/projection,
          /*l_domain=*/dim_domain_from_parallel_tensor_dim_degrees(l_degrees),
          /*r_domain=*/dim_domain_from_parallel_tensor_dim_degrees(r_degrees),
          /*l_dim_ordering=*/get_parallel_tensor_dim_ordering(),
          /*r_dim_ordering=*/get_parallel_tensor_dim_ordering()),
  };
}

ParallelTensorSpaceToParallelTensorSpaceMapping
    invert_parallel_tensor_space_mapping(
        ParallelTensorSpaceToParallelTensorSpaceMapping const &m) {
  return ParallelTensorSpaceToParallelTensorSpaceMapping{
      invert_dim_domain_mapping(m.raw_mapping),
  };
}

} // namespace FlexFlow
