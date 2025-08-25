#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/num_ptensor_shard_dims_t.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/containers/transform.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

num_tensor_dims_t::num_tensor_dims_t(nonnegative_int value_)
  : value(value_)
{ 
  ASSERT(this->value <= MAX_TENSOR_DIM);
}

bool num_tensor_dims_t::operator<(num_tensor_dims_t other) const {
  return this->value < other.value;
}

bool num_tensor_dims_t::operator==(num_tensor_dims_t other) const {
  return this->value == other.value;
}

bool num_tensor_dims_t::operator>(num_tensor_dims_t other) const {
  return this->value > other.value;
}

bool num_tensor_dims_t::operator<=(num_tensor_dims_t other) const {
  return this->value <= other.value;
}

bool num_tensor_dims_t::operator!=(num_tensor_dims_t other) const {
  return this->value != other.value;
}

bool num_tensor_dims_t::operator>=(num_tensor_dims_t other) const {
  return this->value >= other.value;
}

bool num_tensor_dims_t::operator<(nonnegative_int other) const {
  return this->value < other;
}

bool num_tensor_dims_t::operator==(nonnegative_int other) const {
  return this->value == other;
}

bool num_tensor_dims_t::operator>(nonnegative_int other) const {
  return this->value > other;
}

bool num_tensor_dims_t::operator<=(nonnegative_int other) const {
  return this->value <= other;
}

bool num_tensor_dims_t::operator!=(nonnegative_int other) const {
  return this->value != other;
}

bool num_tensor_dims_t::operator>=(nonnegative_int other) const {
  return this->value >= other;
}

bool operator<(nonnegative_int lhs, num_tensor_dims_t rhs) {
  return lhs < rhs.value;
}

bool operator==(nonnegative_int lhs, num_tensor_dims_t rhs) {
  return lhs == rhs.value;
}

bool operator>(nonnegative_int lhs, num_tensor_dims_t rhs) {
  return lhs > rhs.value;
}

bool operator<=(nonnegative_int lhs, num_tensor_dims_t rhs) {
  return lhs <= rhs.value;
}

bool operator!=(nonnegative_int lhs, num_tensor_dims_t rhs) {
  return lhs != rhs.value;
}

bool operator>=(nonnegative_int lhs, num_tensor_dims_t rhs) {
  return lhs >= rhs.value;
}

bool num_tensor_dims_t::operator<(int other) const {
  return this->value < other;
}

bool num_tensor_dims_t::operator==(int other) const {
  return this->value == other;
}

bool num_tensor_dims_t::operator>(int other) const {
  return this->value > other;
}

bool num_tensor_dims_t::operator<=(int other) const {
  return this->value <= other;
}

bool num_tensor_dims_t::operator!=(int other) const {
  return this->value != other;
}

bool num_tensor_dims_t::operator>=(int other) const {
  return this->value >= other;
}
  
bool operator<(int lhs, num_tensor_dims_t rhs) {
  return lhs < rhs.value;
}

bool operator==(int lhs, num_tensor_dims_t rhs) {
  return lhs == rhs.value;
}

bool operator>(int lhs, num_tensor_dims_t rhs) {
  return lhs > rhs.value;
}

bool operator<=(int lhs, num_tensor_dims_t rhs) {
  return lhs <= rhs.value;
}

bool operator!=(int lhs, num_tensor_dims_t rhs) {
  return lhs != rhs.value;
}

bool operator>=(int lhs, num_tensor_dims_t rhs) {
  return lhs >= rhs.value;
}

nonnegative_int num_tensor_dims_t::nonnegative_int_from_num_tensor_dims() const {
  return this->value;
}

int num_tensor_dims_t::int_from_num_tensor_dims() const {
  return this->value.unwrap_nonnegative();
}

void num_tensor_dims_t::check_invariant() const {
  ASSERT(this->value <= MAX_TENSOR_DIM);
}

nonnegative_int format_as(num_tensor_dims_t num_tensor_dims) {
  return num_tensor_dims.nonnegative_int_from_num_tensor_dims();
}

std::ostream &operator<<(std::ostream &s, num_tensor_dims_t num_tensor_dims) {
  return (s << fmt::to_string(num_tensor_dims));
}


num_tensor_dims_t num_tensor_dims_from_num_ptensor_shard_dims(num_ptensor_shard_dims_t num_ptensor_shard_dims) {
  return num_tensor_dims_t{num_ptensor_shard_dims.value};
}

num_tensor_dims_t num_tensor_dims_from_num_ptensor_parallel_dims(num_ptensor_parallel_dims_t num_ptensor_parallel_dims) {
  return num_tensor_dims_from_num_ptensor_shard_dims(
    num_ptensor_shard_dims_from_parallel_dims(
      num_ptensor_parallel_dims));
}

num_ptensor_shard_dims_t num_ptensor_shard_dims_from_num_tensor_dims(num_tensor_dims_t num_tensor_dims) {
  return num_ptensor_shard_dims_t{num_tensor_dims.nonnegative_int_from_num_tensor_dims()};
}

num_ptensor_parallel_dims_t num_ptensor_parallel_dims_from_num_tensor_dims(num_tensor_dims_t num_tensor_dims) {
  return num_ptensor_parallel_dims_from_shard_dims(
    num_ptensor_shard_dims_from_num_tensor_dims(
      num_tensor_dims));
}

std::vector<ff_dim_t> tensor_dims_range(num_tensor_dims_t num_tensor_dims) {
  return transform(nonnegative_range(num_tensor_dims.nonnegative_int_from_num_tensor_dims()),
                   [](nonnegative_int idx) {
                     return ff_dim_t{idx};
                   });
}

std::vector<relative_ff_dim_t> relative_tensor_dims_range(num_tensor_dims_t num_tensor_dims) {
  return transform(nonnegative_range(num_tensor_dims.nonnegative_int_from_num_tensor_dims()),
                   [](nonnegative_int idx) {
                     return relative_ff_dim_t{idx.unwrap_nonnegative()};
                   });
}


} // namespace FlexFlow
