#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H

#include "pcg/machine_specification.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "utils/fmt/vector.h"

namespace FlexFlow {

std::vector<Substitution>
    get_substitution_set(MachineSpecification const &resources);

Substitution create_replicate_linear_combine(positive_int num_dims,
                                             positive_int degree,
                                             bool use_bias);
Substitution create_partition_linear_combine(positive_int num_dims,
                                             positive_int degree,
                                             bool use_bias);
Substitution create_partition_conv2d_combine(positive_int num_dims,
                                             positive_int degree);
Substitution create_partition_attention_combine(positive_int num_heads,
                                                positive_int degree);
Substitution create_replicate_attention_reduce(positive_int num_heads,
                                               positive_int degree);
Substitution create_partition_add_combine(ff_dim_t parallel_dim,
                                          positive_int degree);
Substitution create_partition_relu_combine(ff_dim_t parallel_dim,
                                           positive_int degree);
Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              positive_int degree);
Substitution create_fuse_linear_activation(Activation activation);

} // namespace FlexFlow

#endif
