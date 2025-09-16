#include "pcg/machine_view.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space.h"
#include "pcg/machine_compute_specification.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification_dimension.dtg.h"
#include "pcg/machine_view_dimension.dtg.h"
#include "pcg/stride_t.dtg.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/contains.h"
#include "utils/containers/count.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/scanl.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip3_strict.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

nonnegative_int mv_get_expected_task_space_num_dims(MachineView const &mv) {
  return num_elements(get_strides(mv));
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
  std::vector<MachineViewDimension> dimensions = zip_with_strict(
      strides, dims, [](stride_t s, MachineSpecificationDimension d) {
        return MachineViewDimension{s, d};
      });
  return MachineView{start, dimensions};
}

MachineSpaceCoordinate get_machine_space_coordinate(
    OperatorTaskSpace const &task_space,
    MachineView const &machine_view,
    TaskSpaceCoordinate const &coord) {

  ASSERT(mv_get_expected_task_space_num_dims(machine_view) == op_task_space_num_dims(task_space),
         "Dimension of MachineView must match dimension of OperatorTaskSpace",
         machine_view,
         task_space);
  ASSERT(op_task_space_num_dims(task_space) == task_space_coord_num_dims(coord));
  ASSERT(operator_task_space_contains_coord(task_space, coord));

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

        std::vector<positive_int> sizes =
            transform(dimension_indices, [&](nonnegative_int i) {
              return (
                task_space.degrees.dims.at(i.unwrap_nonnegative()) *
                mv_strides.at(i.unwrap_nonnegative()).unwrapped
              ).positive_int_from_int_ge_two();
            });
        std::vector<nonnegative_int> coord_points =
            transform(dimension_indices, [&](nonnegative_int i) {
              return coord.orthotope_coord.raw.at(i.unwrap_nonnegative());
            });
        std::vector<positive_int> strides =
            transform(dimension_indices, [&](nonnegative_int i) {
              return mv_strides.at(i.unwrap_nonnegative()).unwrapped;
            });

        std::vector<positive_int> coeffs =
            scanl(sizes, 1_p, std::multiplies<positive_int>());

        nonnegative_int index = start_idx;
        for (auto [coeff, coord_point, stride] :
             zip3(coeffs, coord_points, strides)) {
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

  return ms_coord;
}


OperatorSpaceToMachineSpaceMapping
  get_coordinate_mapping_for_machine_view(
    OperatorTaskSpace const &operator_task_space,
    MachineView const &machine_view) {

  return OperatorSpaceToMachineSpaceMapping{
    /*raw_mapping=*/generate_bidict(
      get_task_space_coordinates(operator_task_space),
      [&](TaskSpaceCoordinate const &task_space_coord) {
        return get_machine_space_coordinate(
          /*operator_task_space=*/operator_task_space,
          /*machine_view=*/machine_view,
          /*task_space_coordinate=*/task_space_coord);
      }),
    /*operator_task_space=*/operator_task_space,
  };
}

std::unordered_set<MachineSpaceCoordinate> get_machine_space_coordinates(
    OperatorTaskSpace const &task_space,
    MachineView const &machine_view) {

  ASSERT(op_task_space_num_dims(task_space) == mv_get_expected_task_space_num_dims(machine_view));

  return transform(
      get_task_space_coordinates(task_space), [&](TaskSpaceCoordinate const &coord) {
        return get_machine_space_coordinate(
                task_space, machine_view, coord);
      });
}

std::unordered_set<device_id_t> get_device_ids(OperatorTaskSpace const &task_space,
                                               MachineView const &mv,
                                               MachineComputeSpecification const &ms) {
  ASSERT(op_task_space_num_dims(task_space) == mv_get_expected_task_space_num_dims(mv));

  return transform(get_machine_space_coordinates(task_space, mv),
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
