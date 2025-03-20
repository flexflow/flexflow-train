#ifndef _FLEXFLOW_KERNELS_ARRAY_SHAPE_H
#define _FLEXFLOW_KERNELS_ARRAY_SHAPE_H

#include "kernels/legion_dim.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/stack_vector/stack_vector.h"
#include "utils/visitable.h"
#include <cstddef>
#include <optional>
#include <vector>

namespace FlexFlow {

struct ArrayShape {
public:
  ArrayShape() = delete;
  explicit ArrayShape(LegionOrdered<nonnegative_int> const &dims);

  /**
   * @brief Alias of ArrayShape::num_elements for compatibility with
   * Legion::Domain
   */
  nonnegative_int get_volume() const;

  /**
   * @brief Alias of ArrayShape::num_dims for compatibility with Legion::Domain
   */
  nonnegative_int get_dim() const;

  nonnegative_int num_elements() const;
  nonnegative_int num_dims() const;

  nonnegative_int operator[](legion_dim_t) const;
  nonnegative_int at(legion_dim_t) const;
  nonnegative_int at(ff_dim_t) const;

  bool operator==(ArrayShape const &) const;
  bool operator!=(ArrayShape const &) const;

  legion_dim_t last_idx() const;
  legion_dim_t neg_idx(int) const;

  std::optional<nonnegative_int> at_maybe(legion_dim_t) const;
  std::optional<nonnegative_int> at_maybe(ff_dim_t) const;

  ArrayShape sub_shape(ff_dim_t const &start,
                       std::optional<ff_dim_t> const &end) const;

  ArrayShape sub_shape(legion_dim_t const &start,
                       std::optional<legion_dim_t> const &end) const;

public:
  LegionOrdered<nonnegative_int> dims;

private:
  std::tuple<decltype(dims) const &> tie() const;
};

nonnegative_int get_volume(ArrayShape const &);

ArrayShape array_shape_from_tensor_shape(TensorShape const &);
TensorShape get_tensor_shape(ArrayShape const &, DataType);

std::string format_as(ArrayShape const &);
std::ostream &operator<<(std::ostream &, ArrayShape const &);

} // namespace FlexFlow

#endif
