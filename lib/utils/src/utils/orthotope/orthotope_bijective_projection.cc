#include "utils/orthotope/orthotope_bijective_projection.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/group_by.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/set_of.h"
#include "utils/containers/subvec.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/filter.h"
#include "utils/containers/values.h"
#include "utils/containers/zip_with.h"
#include "utils/exception.h"
#include "utils/orthotope/orthotope.h"
#include "utils/orthotope/orthotope_coordinate.h"
#include "utils/orthotope/orthotope_dim_idx_t.dtg.h"
#include "utils/orthotope/orthotope_dim_idx_t.h"
#include "utils/containers/vector_from_idx_map.h"
#include "utils/containers/scanr.h"
#include "utils/containers/scanr1.h"
#include "utils/containers/all_of.h"
#include "utils/fmt/vector.h"
#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed_from_idx_map.h"

namespace FlexFlow {

OrthotopeBijectiveProjection
  make_orthotope_projection_from_map(std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> const &m, bool reversed) {
    std::unordered_map<int, orthotope_dim_idx_t> raw_idx_map = map_keys(m, [](orthotope_dim_idx_t const &k) { return k.raw_idx; });
    return OrthotopeBijectiveProjection{
      /*dim_mapping=*/vector_from_idx_map(raw_idx_map).value(),
      /*reversed=*/reversed,
    };
}

bool is_valid_projection_between(OrthotopeBijectiveProjection const &proj, Orthotope const &src, Orthotope const &dst) {
  if (proj.reversed) {
    return is_valid_projection_between(reverse_projection(proj), dst, src);
  } 

  auto get_src_dim_size = [&](orthotope_dim_idx_t const &src_idx) -> int {
    return src.dims.at(src_idx);
  };

  auto get_dst_dim_size = [&](orthotope_dim_idx_t const &dst_idx) -> int {
    return dst.dims.at(dst_idx);
  };
  
  std::unordered_map<orthotope_dim_idx_t, std::set<orthotope_dim_idx_t>> src_dims_by_dst_dim
    = get_src_dims_by_dst_dim_map(proj);

  return all_of(src_dims_by_dst_dim, 
                    [&](orthotope_dim_idx_t const &dst_idx, std::set<orthotope_dim_idx_t> const &src_idxs) -> bool {
                      std::vector<int> src_dim_sizes = transform(vector_of(src_idxs), get_src_dim_size);

                      return get_dst_dim_size(dst_idx) == product(src_dim_sizes);
                    });
}
std::unordered_map<orthotope_dim_idx_t, std::set<orthotope_dim_idx_t>> 
  get_src_dims_by_dst_dim_map(OrthotopeBijectiveProjection const &p) {
  if (p.reversed) {
    throw mk_runtime_error(fmt::format("get_src_dims_by_dst_dim_map expected p.reversed=false, but received p={}", p));
  }

  std::set<orthotope_dim_idx_t> src_dim_idxs = dim_idxs_for_orthotope_with_num_dims(get_src_num_dims(p));

  return group_by(src_dim_idxs,
                  [&](orthotope_dim_idx_t const &src_dim_idx) { return get_dst_dim_for_src_dim(p, src_dim_idx); });
}

std::unordered_map<orthotope_dim_idx_t, std::set<orthotope_dim_idx_t>> 
  get_dst_dims_by_src_dim_map(OrthotopeBijectiveProjection const &p) {
  if (!p.reversed) {
    throw mk_runtime_error(fmt::format("get_dst_dims_by_src_dim_map expected p.reversed=true, but received p={}", p));
  }

  std::set<orthotope_dim_idx_t> dst_dim_idxs = dim_idxs_for_orthotope_with_num_dims(get_dst_num_dims(p));

  return group_by(dst_dim_idxs,
                  [&](orthotope_dim_idx_t const &dst_dim_idx) { return get_src_dim_for_dst_dim(p, dst_dim_idx); });
}

std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> get_src_to_dst_dim_map(OrthotopeBijectiveProjection const &p) {
  if (p.reversed) {
    throw mk_runtime_error(fmt::format("get_src_to_dst_dim_map expected p.reversed=false, but received p={}", p));
  }

  std::unordered_map<int, orthotope_dim_idx_t> raw_idx_map = generate_map(range(p.dim_mapping.size()), [&](int x) { return p.dim_mapping.at(x); });
  return map_keys(raw_idx_map, [](int src_dim_idx) { return orthotope_dim_idx_t{src_dim_idx}; });
}

orthotope_dim_idx_t get_dst_dim_for_src_dim(OrthotopeBijectiveProjection const &p, orthotope_dim_idx_t const &src_idx) {
  if (p.reversed) {
    throw mk_runtime_error(fmt::format("get_dst_dim_for_src_dim expected a non-reversed projection, but received: projection={}", p));
  }

  return p.dim_mapping.at(src_idx.raw_idx);
}

orthotope_dim_idx_t get_src_dim_for_dst_dim(OrthotopeBijectiveProjection const &p, orthotope_dim_idx_t const &dst_idx) {
  if (!p.reversed) {
    throw mk_runtime_error(fmt::format("get_src_dim_for_dst_dim expected a reversed projection, but received: projection={}", p));
  }

  return get_dst_dim_for_src_dim(reverse_projection(p), dst_idx);
}

int get_src_num_dims(OrthotopeBijectiveProjection const &p) {
  if (p.reversed) {
    return get_dst_num_dims(reverse_projection(p));
  }

  return p.dim_mapping.size();
}

int get_dst_num_dims(OrthotopeBijectiveProjection const &p) {
  if (p.reversed) {
    return get_src_num_dims(reverse_projection(p));
  } 

  return unordered_set_of(p.dim_mapping).size();
}

OrthotopeBijectiveProjection reverse_projection(OrthotopeBijectiveProjection const &p) {
  OrthotopeBijectiveProjection result = p;
  result.reversed = !p.reversed;
  return result;
}

std::unordered_set<OrthotopeBijectiveProjection> get_all_bijective_projections_between_dim_numbers(int src_num_dims, int dst_num_dims) {
  if (src_num_dims < dst_num_dims) {
    return transform(get_all_bijective_projections_between_dim_numbers(dst_num_dims, src_num_dims), 
                     [](OrthotopeBijectiveProjection const &p) { return reverse_projection(p); });
  }

  std::set<orthotope_dim_idx_t> src_dim_idxs = dim_idxs_for_orthotope_with_num_dims(src_num_dims);
  std::set<orthotope_dim_idx_t> dst_dim_idxs = dim_idxs_for_orthotope_with_num_dims(dst_num_dims);

  std::unordered_map<orthotope_dim_idx_t, std::unordered_set<orthotope_dim_idx_t>> src_to_dst_idxs = 
    generate_map(src_dim_idxs, [&](orthotope_dim_idx_t) { return unordered_set_of(dst_dim_idxs); });

  std::unordered_set<std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t>> valid_mappings =  
    filter(get_all_assignments(src_to_dst_idxs),
           [&](std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> const &src_to_dst_idx) { 
             return set_of(values(src_to_dst_idx)) == dst_dim_idxs;
           });

  return transform(valid_mappings, [](std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> const &m) { return make_orthotope_projection_from_map(m, /*reversed=*/false); });
}

std::unordered_set<OrthotopeBijectiveProjection> get_all_bijective_projections_between(Orthotope const &src, Orthotope const &dst) {
  return filter(get_all_bijective_projections_between_dim_numbers(/*src_num_dims=*/orthotope_num_dims(src), /*dst_num_dims=*/orthotope_num_dims(dst)),
                [&](OrthotopeBijectiveProjection const &p) {
                  return is_valid_projection_between(p, /*src=*/src, /*dst=*/dst);
                });
}

int project_into_1d(Orthotope const &orthotope, OrthotopeCoordinate const &coord) {
  if (!orthotope_contains_coord(orthotope, coord)) {
    throw mk_runtime_error(fmt::format("coord out of bounds of orthotope: orthotope={}, coord={}", orthotope, coord));
  }

  if (orthotope.dims.size() == 0) {
    return 0;
  }

  std::vector<std::pair<int, int>> coords_and_sizes = zip(coord.idxs.get_contents(),
                                                          orthotope.dims.get_contents());

  std::vector<int> coords = transform(coords_and_sizes, [](std::pair<int, int> const &p) { return p.first; });
  std::vector<int> dim_sizes = transform(coords_and_sizes, [](std::pair<int, int> const &p) { return p.second; });

  std::vector<int> strides = scanr(subvec(dim_sizes, 1, std::nullopt), 1, [](int next, int accum) { return accum * next; });
  return sum(zip_with(coords, strides, [](int coord, int stride) { return coord * stride; }));
}

OrthotopeCoordinate project_out_of_1d(int one_dimensional_coord, Orthotope const &dst_orthotope) {
  if (dst_orthotope.dims.size() == 0) {
    if (one_dimensional_coord == 0) {
      return OrthotopeCoordinate{{}};
    } else {
      throw mk_runtime_error(fmt::format("Only valid one_dimensional_coord for zero-dimensional orthotope is 0, but receieved one_dimensional_coord={}", one_dimensional_coord));
    }
  }

  if (one_dimensional_coord >= orthotope_get_volume(dst_orthotope)) {
    throw mk_runtime_error(fmt::format("project_out_of_1d received coordinate that would be out of bounds of dst orthotope: dst_orthotope={}, coordinate={}", dst_orthotope, one_dimensional_coord));
  }

  std::vector<int> dim_sizes = dst_orthotope.dims.get_contents();
  std::vector<int> strides = scanr(subvec(dim_sizes, 1, std::nullopt), 1, [](int next, int accum) { return accum * next; });

  OrthotopeCoordinate result = OrthotopeCoordinate{
    orthotope_dim_indexed_of(zip_with(dim_sizes, strides, [&](int dim_size, int stride) { return (one_dimensional_coord / stride) % dim_size; })),
  };
  return result;
}

OrthotopeCoordinate project_coordinate_through(OrthotopeBijectiveProjection const &p, Orthotope const &src_orthotope, OrthotopeCoordinate const &src_coord, Orthotope const &dst_orthotope) {
  std::set<orthotope_dim_idx_t> dst_dim_idxs = get_orthotope_dims(dst_orthotope);
  std::set<orthotope_dim_idx_t> src_dim_idxs = get_orthotope_dims(src_orthotope);

  if (src_coord.idxs.size() != get_src_num_dims(p)) {
    throw mk_runtime_error(fmt::format("project_coordinate_through requires projection src and coordinate to have same num dims, but got {} and {} respectively",
                                       get_src_num_dims(p),
                                       src_coord.idxs.size()));
  }

  if (!orthotope_contains_coord(src_orthotope, src_coord)) {
    throw mk_runtime_error(fmt::format("project_coordinate_through requires coord to be in the orthotope, but got coord={} and orthotope={} respectively", src_coord, src_orthotope));
  }

  if (p.reversed) {
    std::unordered_map<orthotope_dim_idx_t, 
                       std::set<orthotope_dim_idx_t>> 
      dst_dim_idxs_by_src_dim_idx = 
      group_by(dst_dim_idxs,
               [&](orthotope_dim_idx_t const &dst_dim_idx) { return get_src_dim_for_dst_dim(p, dst_dim_idx); });


    std::unordered_map<orthotope_dim_idx_t, Orthotope> dst_sub_orthotopes_by_src_dim_idx = 
      map_values(dst_dim_idxs_by_src_dim_idx, 
                 [&](std::set<orthotope_dim_idx_t> const &dst_dim_idxs) {
                   return orthotope_drop_dims_except(dst_orthotope, dst_dim_idxs);
                 });

    std::unordered_map<orthotope_dim_idx_t, OrthotopeCoordinate> dst_coords_by_src_dim_idx =
      generate_map(src_dim_idxs,
                   [&](orthotope_dim_idx_t const &src_idx) -> OrthotopeCoordinate {
                     return project_out_of_1d(src_coord.idxs.at(src_idx),
                                              dst_sub_orthotopes_by_src_dim_idx.at(src_idx));
                   });

    std::unordered_map<orthotope_dim_idx_t, int> dst_coords = merge_maps(
      transform(vector_of(src_dim_idxs), [&](orthotope_dim_idx_t const &src_idx) -> std::unordered_map<orthotope_dim_idx_t, int> {
                           return map_from_keys_and_values(
                             vector_of(dst_dim_idxs_by_src_dim_idx.at(src_idx)),
                             dst_coords_by_src_dim_idx.at(src_idx).idxs.get_contents());
                         }));

    return OrthotopeCoordinate{
      orthotope_dim_indexed_from_idx_map(dst_coords).value(),
    };
  } else {
    std::unordered_map<orthotope_dim_idx_t, std::set<orthotope_dim_idx_t>> src_dim_idxs_by_dst_dim_idx = 
      group_by(src_dim_idxs,
               [&](orthotope_dim_idx_t const &src_dim_idx) { return get_dst_dim_for_src_dim(p, src_dim_idx); });

    
    std::unordered_map<orthotope_dim_idx_t, Orthotope> src_sub_orthotopes_by_dst_dim_idx = 
      map_values(src_dim_idxs_by_dst_dim_idx, 
                 [&](std::set<orthotope_dim_idx_t> const &src_dim_idxs) {
                   return orthotope_drop_dims_except(src_orthotope, src_dim_idxs);
                 });

    std::unordered_map<orthotope_dim_idx_t, OrthotopeCoordinate> src_sub_coords_by_dst_dim_idx = 
      map_values(src_dim_idxs_by_dst_dim_idx, 
                 [&](std::set<orthotope_dim_idx_t> const &src_dim_idxs) {
                   return orthotope_coord_drop_dims_except(src_coord, src_dim_idxs);
                 });

    std::unordered_map<orthotope_dim_idx_t, int> dst_coords = 
      generate_map(dst_dim_idxs, 
                   [&](orthotope_dim_idx_t const &dst_idx) {
                     return project_into_1d(
                        src_sub_orthotopes_by_dst_dim_idx.at(dst_idx),
                        src_sub_coords_by_dst_dim_idx.at(dst_idx));
                   });


    return OrthotopeCoordinate{
      orthotope_dim_indexed_from_idx_map(dst_coords).value(),
    };
  }
}

} // namespace FlexFlow
