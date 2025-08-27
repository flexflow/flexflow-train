#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"

namespace FlexFlow {

ParallelTensorSpaceToParallelTensorSpaceMapping
  parallel_tensor_space_mapping_from_projection(
    DimProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t> const &projection, 
    ParallelTensorDimDegrees const &l_degrees,
    ParallelTensorDimDegrees const &r_degrees) {

  return ParallelTensorSpaceToParallelTensorSpaceMapping{
    dim_domain_mapping_from_projection(
      /*projection=*/projection,
      /*l_domain=*/dim_domain_from_parallel_tensor_dim_degrees(l_degrees),
      /*r_domain=*/dim_domain_from_parallel_tensor_dim_degrees(r_degrees),
      /*l_dim_ordering=*/get_parallel_tensor_dim_ordering(),
      /*r_dim_ordering=*/get_parallel_tensor_dim_ordering()),
  };
}


} // namespace FlexFlow
