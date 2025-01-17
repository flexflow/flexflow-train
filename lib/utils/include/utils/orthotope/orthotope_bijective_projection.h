#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BIJECTIVE_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BIJECTIVE_PROJECTION_H

#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope_bijective_projection.dtg.h"
#include "utils/orthotope/orthotope_coordinate.dtg.h"
#include <unordered_set>
#include <set>

namespace FlexFlow {

OrthotopeBijectiveProjection
  make_orthotope_projection_from_map(std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> const &, bool reversed);

bool is_valid_projection_between(OrthotopeBijectiveProjection const &proj, Orthotope const &src, Orthotope const &dst);

std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> get_src_to_dst_dim_map(OrthotopeBijectiveProjection const &);
std::unordered_map<orthotope_dim_idx_t, std::set<orthotope_dim_idx_t>> get_dst_dims_by_src_dim_map(OrthotopeBijectiveProjection const &);
std::unordered_map<orthotope_dim_idx_t, std::set<orthotope_dim_idx_t>> get_src_dims_by_dst_dim_map(OrthotopeBijectiveProjection const &);

orthotope_dim_idx_t get_dst_dim_for_src_dim(OrthotopeBijectiveProjection const &, orthotope_dim_idx_t const &);
orthotope_dim_idx_t get_src_dim_for_dst_dim(OrthotopeBijectiveProjection const &, orthotope_dim_idx_t const &);

int get_src_num_dims(OrthotopeBijectiveProjection const &);
int get_dst_num_dims(OrthotopeBijectiveProjection const &);

OrthotopeBijectiveProjection reverse_projection(OrthotopeBijectiveProjection const &);

std::unordered_set<OrthotopeBijectiveProjection> get_all_bijective_projections_between(Orthotope const &src, Orthotope const &dst);
std::unordered_set<OrthotopeBijectiveProjection> get_all_bijective_projections_between_dim_numbers(int src_num_dims, int dst_num_dims);

int project_into_1d(Orthotope const &, OrthotopeCoordinate const &);
OrthotopeCoordinate project_out_of_1d(int, Orthotope const &);


OrthotopeCoordinate project_coordinate_through(OrthotopeBijectiveProjection const &projection, 
                                               Orthotope const &src_orthotope, 
                                               OrthotopeCoordinate const &src_coord, 
                                               Orthotope const &dst_orthotope);

} // namespace FlexFlow

#endif
