#include "pcg/machine_view.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_specification_dimension.dtg.h"
#include "pcg/machine_view_dimension.dtg.h"
#include "pcg/operator_task_space.dtg.h"
#include "pcg/operator_task_space.h"
#include "pcg/stride_t.dtg.h"
#include "utils/containers/contains.h"
#include "utils/containers/count.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/scanl.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

size_t num_dims(MachineView const &mv) {
  return get_strides(mv).size();
}

DeviceType get_device_type(MachineView const &mv) {
  return mv.start.device_type;
}

std::vector<stride_t> get_strides(MachineView const &mv) {
  return transform(mv.dimensions,
                   [](MachineViewDimension const &dim) { return dim.stride; });
}

std::vector<MachineSpecificationDimension>
    get_dimensions(MachineView const &mv) {
  return transform(mv.dimensions, [](MachineViewDimension const &dim) {
    return dim.projection;
  });
}

MachineView machine_view_from_strides_and_machine_spec_dimensions(
    MachineSpaceCoordinate const &start,
    std::vector<stride_t> const &strides,
    std::vector<MachineSpecificationDimension> const &dims) {
  if (strides.size() != dims.size()) {
    throw mk_runtime_error(fmt::format(
        "Length of strides ({}) and dims ({}) must match when calling "
        "machine_view_from_strides_and_machine_spec_dimensions",
        start,
        strides));
  }
  std::vector<MachineViewDimension> dimensions =
      transform(zip(strides, dims), [&](auto const &p) {
        return MachineViewDimension{p.first, p.second};
      });
  return MachineView{start, dimensions};
}

std::optional<MachineSpaceCoordinate> get_machine_space_coordinate(
    OperatorTaskSpace const &task,
    MachineView const &machine_view,
    TaskSpaceCoordinate const &coord,
    MachineSpecification const &machine_specification) {

  if (num_dims(machine_view) != task.degrees.size()) {
    throw mk_runtime_error(
        fmt::format("Dimension of machine_view ({}) must match dimension of "
                    "task ({}) when computing machine space coordinate",
                    machine_view,
                    task.degrees));
  }

  auto get_dimension_indices_for_dimension =
      [&](MachineSpecificationDimension dimension)
      -> std::vector<nonnegative_int> {
    std::vector<MachineSpecificationDimension> mv_dimensions =
        get_dimensions(machine_view);
    return filter(nonnegative_range(num_elements(mv_dimensions)),
                  [&](nonnegative_int idx) {
                    return mv_dimensions.at(idx.unwrap_nonnegative()) ==
                           dimension;
                  });
  };

  auto compute_index =
      [&](nonnegative_int start_idx,
          std::vector<nonnegative_int> const &dimension_indices) {
        std::vector<stride_t> mv_strides = get_strides(machine_view);

        std::vector<nonnegative_int> sizes =
            transform(dimension_indices, [&](nonnegative_int i) {
              return task.degrees.at(i.unwrap_nonnegative()) *
                     mv_strides.at(i.unwrap_nonnegative()).unwrapped;
            });
        std::vector<nonnegative_int> coord_points =
            transform(dimension_indices, [&](nonnegative_int i) {
              return coord.raw_coord.at(i.unwrap_nonnegative());
            });
        std::vector<nonnegative_int> strides =
            transform(dimension_indices, [&](nonnegative_int i) {
              return mv_strides.at(i.unwrap_nonnegative()).unwrapped;
            });

        std::vector<nonnegative_int> coeffs = scanl(
            sizes, nonnegative_int{1}, std::multiplies<nonnegative_int>());

        nonnegative_int index = start_idx;
        for (auto [coeff, coord_point, stride] :
             zip(coeffs, coord_points, strides)) {
          index += coeff * coord_point * stride;
        }
        return index;
      };

  std::vector<nonnegative_int> inter_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTER_NODE);
  std::vector<nonnegative_int> intra_dimension_indices =
      get_dimension_indices_for_dimension(
          MachineSpecificationDimension::INTRA_NODE);

  nonnegative_int node_idx =
      compute_index(machine_view.start.node_idx, inter_dimension_indices);
  nonnegative_int device_idx =
      compute_index(machine_view.start.device_idx, intra_dimension_indices);
  MachineSpaceCoordinate ms_coord = MachineSpaceCoordinate{
      node_idx, device_idx, get_device_type(machine_view)};

  if (!is_valid_machine_space_coordinate(machine_specification, ms_coord)) {
    return std::nullopt;
  }
  return ms_coord;
}

std::unordered_set<MachineSpaceCoordinate> get_machine_space_coordinates(
    OperatorTaskSpace const &task,
    MachineView const &machine_view,
    MachineSpecification const &machine_specification) {
  return transform(
      get_task_space_coordinates(task), [&](TaskSpaceCoordinate const &coord) {
        std::optional<MachineSpaceCoordinate> maybe_coordinate =
            get_machine_space_coordinate(
                task, machine_view, coord, machine_specification);
        if (!maybe_coordinate.has_value()) {
          throw mk_runtime_error(
              fmt::format("In get_machine_space_coordinates, the given "
                          "OperatorTaskSpace {} and MachineView {} are not "
                          "compatible with the given MachineSpecification {}",
                          task,
                          machine_view,
                          machine_specification));
        }
        return maybe_coordinate.value();
      });
}

std::unordered_set<device_id_t> get_device_ids(OperatorTaskSpace const &task,
                                               MachineView const &mv,
                                               MachineSpecification const &ms) {
  return transform(get_machine_space_coordinates(task, mv, ms),
                   [&](MachineSpaceCoordinate const &coord) {
                     return get_device_id(ms, coord);
                   });
}

MachineView make_1d_machine_view(MachineSpaceCoordinate const &start,
                                 MachineSpecificationDimension const &dim,
                                 stride_t stride) {

  return machine_view_from_strides_and_machine_spec_dimensions(
      start, {stride}, {dim});
}

} // namespace FlexFlow
