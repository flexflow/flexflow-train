#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H

#include "pcg/machine_specification.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "utils/fmt/vector.h"

namespace FlexFlow {

std::vector<Substitution>
    get_substitution_set(MachineSpecification const &resources);

Substitution create_combine_inception(int num_convs, int num_dims, int degree);
Substitution create_combine_concat(int num_inputs, int num_dims, int degree);
Substitution
    create_replicate_linear_combine(int num_dims, int degree, bool use_bias);
Substitution create_partition_linear_combine(int num_dims,
                                             int degree,
                                             Activation activation,
                                             bool use_bias);
Substitution create_partition_conv2d_combine(int num_dims, int degree);
Substitution create_partition_attention_combine(int num_heads, int degree);
Substitution create_replicate_attention_reduce(int num_heads, int degree);
Substitution create_partition_add_combine(ff_dim_t parallel_dim, int degree);
Substitution create_partition_relu_combine(ff_dim_t parallel_dim, int degree);
Substitution create_partition_concat_combine(int num_inputs,
                                             ff_dim_t concat_dim,
                                             ff_dim_t parallel_dim,
                                             int degree);
Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              int degree);
Substitution create_fuse_linear_activation(Activation activation);

} // namespace FlexFlow

#endif
