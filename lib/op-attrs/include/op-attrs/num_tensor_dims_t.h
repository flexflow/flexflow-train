#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_NUM_TENSOR_DIMS_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_NUM_TENSOR_DIMS_T_H

#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/relative_ff_dim_t.dtg.h"

namespace FlexFlow {

struct num_tensor_dims_t {
public:
  num_tensor_dims_t() = delete;
  num_tensor_dims_t(nonnegative_int);

  bool operator<(num_tensor_dims_t other) const;
  bool operator==(num_tensor_dims_t other) const;
  bool operator>(num_tensor_dims_t other) const;
  bool operator<=(num_tensor_dims_t other) const;
  bool operator!=(num_tensor_dims_t other) const;
  bool operator>=(num_tensor_dims_t other) const;

  bool operator<(nonnegative_int other) const;
  bool operator==(nonnegative_int other) const;
  bool operator>(nonnegative_int other) const;
  bool operator<=(nonnegative_int other) const;
  bool operator!=(nonnegative_int other) const;
  bool operator>=(nonnegative_int other) const;

  friend bool operator<(nonnegative_int lhs, num_tensor_dims_t rhs);
  friend bool operator==(nonnegative_int lhs, num_tensor_dims_t rhs);
  friend bool operator>(nonnegative_int lhs, num_tensor_dims_t rhs);
  friend bool operator<=(nonnegative_int lhs, num_tensor_dims_t rhs);
  friend bool operator!=(nonnegative_int lhs, num_tensor_dims_t rhs);
  friend bool operator>=(nonnegative_int lhs, num_tensor_dims_t rhs);

  bool operator<(int other) const;
  bool operator==(int other) const;
  bool operator>(int other) const;
  bool operator<=(int other) const;
  bool operator!=(int other) const;
  bool operator>=(int other) const;

  friend bool operator<(int lhs, num_tensor_dims_t rhs);
  friend bool operator==(int lhs, num_tensor_dims_t rhs);
  friend bool operator>(int lhs, num_tensor_dims_t rhs);
  friend bool operator<=(int lhs, num_tensor_dims_t rhs);
  friend bool operator!=(int lhs, num_tensor_dims_t rhs);
  friend bool operator>=(int lhs, num_tensor_dims_t rhs);

  nonnegative_int nonnegative_int_from_num_tensor_dims() const;
  int int_from_num_tensor_dims() const;

private:
  nonnegative_int value;

private:
  void check_invariant() const;
};

nonnegative_int format_as(num_tensor_dims_t);
std::ostream &operator<<(std::ostream &, num_tensor_dims_t);

num_tensor_dims_t 
  num_tensor_dims_from_num_ptensor_shard_dims(num_ptensor_shard_dims_t);

num_tensor_dims_t num_tensor_dims_from_num_ptensor_parallel_dims(num_ptensor_parallel_dims_t);

num_ptensor_shard_dims_t num_ptensor_shard_dims_from_num_tensor_dims(num_tensor_dims_t);

num_ptensor_parallel_dims_t num_ptensor_parallel_dims_from_num_tensor_dims(num_tensor_dims_t);

std::vector<ff_dim_t> tensor_dims_range(num_tensor_dims_t);
std::vector<relative_ff_dim_t> relative_tensor_dims_range(num_tensor_dims_t);

} // namespace FlexFlow

#endif
