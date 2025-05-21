#include "op-attrs/replica_parallel_dim_set.h"
#include "utils/exception.h"

namespace FlexFlow {

ReplicaParallelDimSet empty_replica_parallel_dim_set() {
  return ReplicaParallelDimSet{SumDegree{1_p}, DiscardCopyDegree{1_p}};
}

positive_int get_degree_of_replica_type(ReplicaParallelDimSet const &s,
                                           ReplicaType replica_type) {
  switch (replica_type) {
    case ReplicaType::SUM:
      return s.sum_degree.value;
    case ReplicaType::DISCARD_COPY:
      return s.discard_copy_degree.value;
    default:
      throw mk_runtime_error(fmt::format("Unexpected ReplicaType value: {}",
                                         static_cast<int>(replica_type)));
  }
}

std::unordered_set<ReplicaParallelDim>
    get_replica_dims(ReplicaParallelDimSet const &s) {
  return std::unordered_set<ReplicaParallelDim>{
      ReplicaParallelDim{s.sum_degree.value, ReplicaType::SUM},
      ReplicaParallelDim{s.discard_copy_degree.value,
                         ReplicaType::DISCARD_COPY},
  };
}

} // namespace FlexFlow
