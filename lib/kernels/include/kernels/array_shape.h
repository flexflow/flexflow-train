#ifndef _FLEXFLOW_KERNELS_ARRAY_SHAPE_H
#define _FLEXFLOW_KERNELS_ARRAY_SHAPE_H

#include "kernels/array_coord.dtg.h"
#include "kernels/legion_dim.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/positive_int/positive_int.h"
#include "utils/stack_vector/stack_vector.h"
#include "utils/visitable.h"
#include <cstddef>
#include <optional>
#include <vector>

namespace FlexFlow {

struct ArrayShape {
public:
  ArrayShape() = delete;
  explicit ArrayShape(LegionOrdered<positive_int> const &dims);

  positive_int num_elements() const;

  nonnegative_int num_dims() const;

  positive_int operator[](legion_dim_t) const;
  positive_int at(legion_dim_t) const;
  positive_int at(ff_dim_t) const;

  bool operator==(ArrayShape const &) const;
  bool operator!=(ArrayShape const &) const;

  legion_dim_t last_idx() const;
  legion_dim_t neg_idx(int) const;

  std::optional<positive_int> at_maybe(legion_dim_t) const;
  std::optional<positive_int> at_maybe(ff_dim_t) const;

  ArrayShape sub_shape(ff_dim_t const &start,
                       std::optional<ff_dim_t> const &end) const;

  ArrayShape sub_shape(legion_dim_t const &start,
                       std::optional<legion_dim_t> const &end) const;

public:
  LegionOrdered<positive_int> dims;

private:
  std::tuple<decltype(dims) const &> tie() const;

  friend ::std::hash<ArrayShape>;
};

std::string format_as(ArrayShape const &);
std::ostream &operator<<(std::ostream &, ArrayShape const &);

positive_int get_num_elements(ArrayShape const &);

ArrayShape array_shape_from_tensor_shape(TensorShape const &);
TensorShape tensor_shape_from_array_shape(ArrayShape const &, DataType);
TensorDims tensor_dims_from_array_shape(ArrayShape const &);

std::unordered_set<ff_dim_t> get_ff_dim_t_set(ArrayShape const &);
std::unordered_set<ArrayCoord> get_array_coord_set(ArrayShape const &);

ArrayShape
    array_shape_drop_dims(ArrayShape const &shape,
                          std::function<bool(ff_dim_t)> const &should_drop_dim);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::ArrayShape> {
  size_t operator()(::FlexFlow::ArrayShape const &) const;
};

} // namespace std

#endif
