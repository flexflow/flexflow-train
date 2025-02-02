#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/dim_ordered/transform.h"
#include "op-attrs/dim_ordered/zip.h"
#include "op-attrs/replica_parallel_dim.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/all_of.h"
#include "utils/containers/product.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

FFOrdered<ShardParallelDim> ff_ordered_shard_dims(ParallelTensorDims const &d) {
  return d.shard_dims;
}

FFOrdered<nonnegative_int>
    ff_ordered_shard_degrees(ParallelTensorDims const &d) {
  return transform(d.shard_dims,
                   [](ShardParallelDim const &d) { return d.degree; });
}

std::unordered_set<ReplicaParallelDim>
    replica_dims(ParallelTensorDims const &d) {
  return get_replica_dims(d.replica_dims);
}

nonnegative_int num_shard_dims(ParallelTensorDims const &dims) {
  return num_elements(dims.shard_dims);
}

ParallelTensorDimDegrees get_parallel_degrees(ParallelTensorDims const &d) {
  return ParallelTensorDimDegrees{
      d.replica_dims.sum_degree,
      d.replica_dims.discard_copy_degree,
      ff_ordered_shard_degrees(d),
  };
}

ParallelTensorDims lift_to_parallel(TensorDims const &dims) {
  std::vector<nonnegative_int> shard_degrees =
      repeat_element(/*num_times=*/num_dims(dims), /*element=*/1_n);
  return lift_to_parallel_with_degrees(
      dims, SumDegree{1_n}, DiscardCopyDegree{1_n}, shard_degrees);
}

ParallelTensorDims lift_to_parallel_with_degrees(
    TensorDims const &unpar,
    SumDegree const &sum_degree,
    DiscardCopyDegree const &discard_copy_degree,
    FFOrdered<nonnegative_int> const &shard_degrees) {
  std::vector<ShardParallelDim> lifted =
      transform(zip(vector_of(unpar.ff_ordered), vector_of(shard_degrees)),
                [](std::pair<nonnegative_int, nonnegative_int> const &p) {
                  nonnegative_int size = p.first;
                  nonnegative_int degree = p.second;
                  return ShardParallelDim{size, degree};
                });

  return ParallelTensorDims{FFOrdered<ShardParallelDim>{lifted},
                            ReplicaParallelDimSet{
                                sum_degree,
                                discard_copy_degree,
                            }};
}

ParallelTensorDims
    lift_to_parallel_with_degrees(TensorDims const &unpar,
                                  ParallelTensorDimDegrees const &degrees) {
  return lift_to_parallel_with_degrees(unpar,
                                       degrees.sum_degree,
                                       degrees.discard_copy_degree,
                                       degrees.shard_degrees);
}

nonnegative_int total_replica_degree(ParallelTensorDims const &dims) {
  return dims.replica_dims.discard_copy_degree.value *
         dims.replica_dims.sum_degree.value;
}

nonnegative_int total_shard_degree(ParallelTensorDims const &dims) {
  return product(transform(vector_of(dims.shard_dims),
                           [](ShardParallelDim const &d) { return d.degree; }));
}

nonnegative_int total_parallel_degree(ParallelTensorDims const &dims) {
  return total_replica_degree(dims) * total_shard_degree(dims);
}

bool is_valid(ParallelTensorDims const &dims) {
  return all_of(dims.shard_dims,
                [](ShardParallelDim const &d) { return is_valid(d); }) &&
         all_of(replica_dims(dims),
                [](ReplicaParallelDim const &d) { return is_valid(d); });
}

ShardParallelDim shard_dim_at_idx(ParallelTensorDims const &d,
                                  relative_ff_dim_t idx) {
  return d.shard_dims.at(idx);
}

ShardParallelDim &shard_dim_at_idx(ParallelTensorDims &d,
                                   relative_ff_dim_t idx) {
  return d.shard_dims.at(idx);
}

TensorDims get_piece_dims(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_reduced_dims(ParallelTensorDims const &dims) {
  FFOrdered<nonnegative_int> dim_sizes = transform(
      dims.shard_dims, [](ShardParallelDim const &d) { return d.size; });
  return TensorDims{dim_sizes};
}

} // namespace FlexFlow
