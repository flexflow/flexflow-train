#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PARALLEL_TENSOR_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PARALLEL_TENSOR_SPACE_MAPPING_H

#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "utils/bidict/bidict.h"
#include "utils/bidict/algorithms/left_entries.h"
#include <libassert/assert.hpp>
#include "utils/hash/tuple.h"

namespace FlexFlow {

template <typename T>
struct ParallelTensorSpaceMapping {
public:
  ParallelTensorSpaceMapping() = delete;
  ParallelTensorSpaceMapping(
    ParallelTensorDimDegrees const &domain,
    bidict<ParallelTensorSpaceCoordinate, T> const &coord_map) 
    : domain(domain), coord_map(coord_map) 
  {
    ASSERT(get_parallel_tensor_space_coordinates(domain) == left_entries(coord_map));
  }

  ParallelTensorDimDegrees const &get_domain() const {
    return this->domain;
  }

  bidict<ParallelTensorSpaceCoordinate, T> const &get_coord_map() const {
    return this->coord_map;
  }
private:
  ParallelTensorDimDegrees domain;
  bidict<ParallelTensorSpaceCoordinate, T> coord_map;
private:
  std::tuple<
    decltype(domain) const &,
    decltype(coord_map) const &
  > tie() const {
    return std::tie(this->domain, this->coord_map);
  }

public:
  bool operator==(ParallelTensorSpaceMapping const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(ParallelTensorSpaceMapping const &other) const {
    return this->tie() != other.tie();
  }

  bool operator<(ParallelTensorSpaceMapping const &other) const {
    return this->tie() < other.tie();
  }

  bool operator>(ParallelTensorSpaceMapping const &other) const {
    return this->tie() > other.tie();
  }

  bool operator<=(ParallelTensorSpaceMapping const &other) const {
    return this->tie() <= other.tie();
  }

  bool operator>=(ParallelTensorSpaceMapping const &other) const {
    return this->tie() >= other.tie();
  }

  friend struct std::hash<ParallelTensorSpaceMapping<T>>;
};

template <typename T>
std::string format_as(ParallelTensorSpaceMapping<T> const &mapping) {
  return fmt::format("<ParallelTensorSpaceMapping domain={} coord_map={}>", mapping.get_domain(), mapping.get_coord_map());
}

template <typename T>
std::ostream &operator<<(std::ostream &s, ParallelTensorSpaceMapping<T> const &mapping) {
  return (s << fmt::to_string(mapping));
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::ParallelTensorSpaceMapping<T>> {
  size_t operator()(::FlexFlow::ParallelTensorSpaceMapping<T> const &mapping) {
    return get_std_hash(mapping.tie());
  }
};

}

#endif
