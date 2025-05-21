#include "kernels/create_accessor_with_contents.h"

namespace FlexFlow {

template GenericTensorAccessorW
    create_1d_accessor_w_with_contents(std::vector<bool> const &, Allocator &);

template GenericTensorAccessorW
    create_2d_accessor_w_with_contents(std::vector<std::vector<bool>> const &,
                                       Allocator &);

template GenericTensorAccessorW create_3d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<bool>>> const &, Allocator &);

template GenericTensorAccessorW create_4d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<std::vector<bool>>>> const &,
    Allocator &);

template GenericTensorAccessorR
    create_1d_accessor_r_with_contents(std::vector<bool> const &, Allocator &);

template GenericTensorAccessorR
    create_2d_accessor_r_with_contents(std::vector<std::vector<bool>> const &,
                                       Allocator &);

template GenericTensorAccessorR create_3d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<bool>>> const &, Allocator &);

template GenericTensorAccessorR create_4d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<std::vector<bool>>>> const &,
    Allocator &);

} // namespace FlexFlow
