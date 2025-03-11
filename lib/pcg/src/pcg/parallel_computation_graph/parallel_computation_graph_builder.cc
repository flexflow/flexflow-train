#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/ops/batch_matmul_attrs.dtg.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/ops/combine_attrs.dtg.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/parallel_op_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/shape_inference.h"
#include "pcg/parallel_computation_graph/generate_weight_transform.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/count.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_with.h"

namespace FlexFlow {

static std::string get_default_name(OperatorType op_type) {
  return get_operator_type_name(op_type);
}

static std::string get_default_name(PCGOperatorAttrs const &attrs) {
  return get_default_name(get_op_type(attrs));
}

ParallelComputationGraphBuilder::ParallelComputationGraphBuilder()
    : pcg(empty_parallel_computation_graph()) {}

parallel_tensor_guid_t ParallelComputationGraphBuilder::create_input_tensor(
    TensorShape const &shape, std::optional<std::string> const &name) {

  ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{InputAttrs{shape}},
      name,
  };

  return get_only(
      add_parallel_layer(
          this->pcg, layer_attrs, {}, {}, std::vector{CreateGrad::NO})
          .outputs);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add(
    parallel_tensor_guid_t const &lhs,
    parallel_tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {

  ParallelTensorShape lhs_shape = this->get_shape(lhs);
  ParallelTensorShape rhs_shape = this->get_shape(rhs);

  DataType datatype = [&] {
    if (lhs_shape.data_type != rhs_shape.data_type) {
      throw mk_runtime_error(
          fmt::format("Datatypes do not match: {} (lhs) != {} (rhs)",
                      lhs_shape.data_type,
                      rhs_shape.data_type));
    } else {
      return lhs_shape.data_type;
    }
  }();

  ElementBinaryAttrs attrs = ElementBinaryAttrs{
      OperatorType::EW_ADD,
      datatype,
      false,
      false,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {lhs, rhs}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::batch_matmul(
    parallel_tensor_guid_t const &a,
    parallel_tensor_guid_t const &b,
    std::optional<std::string> const &maybe_name) {

  BatchMatmulAttrs attrs = BatchMatmulAttrs{
      /*a_seq_length_dim=*/std::nullopt,
      /*b_seq_length_dim=*/std::nullopt,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {a, b}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::cast(
    parallel_tensor_guid_t const &input,
    DataType result_type,
    std::optional<std::string> const &maybe_name) {

  CastAttrs attrs = CastAttrs{result_type};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {input}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::conv2d(
    parallel_tensor_guid_t const &raw_input,
    nonnegative_int outChannels,
    nonnegative_int kernelH,
    nonnegative_int kernelW,
    nonnegative_int strideH,
    nonnegative_int strideW,
    nonnegative_int paddingH,
    nonnegative_int paddingW,
    std::optional<Activation> const &activation,
    nonnegative_int groups,
    bool use_bias,
    std::optional<InitializerAttrs> const &maybe_kernel_initializer,
    std::optional<InitializerAttrs> const &maybe_bias_initializer,
    std::optional<RegularizerAttrs> const &kernel_regularizer,
    std::optional<std::string> const &maybe_name) {
  Conv2DAttrs attrs = Conv2DAttrs{
      /*out_channels=*/outChannels,
      /*kernel_h=*/kernelH,
      /*kernel_w=*/kernelW,
      /*stride_h=*/strideH,
      /*stride_w=*/strideW,
      /*padding_h=*/paddingH,
      /*padding_w=*/paddingW,
      /*groups=*/groups,
      /*activation=*/activation,
      /*use_bias=*/use_bias,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  parallel_tensor_guid_t input =
      this->as_type(raw_input, DataType::FLOAT, name + "input_pre_cast");

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::vector<InitializerAttrs> initializers =
      get_initializers(attrs,
                       get_reduced_shape(input_shape),
                       maybe_kernel_initializer,
                       maybe_bias_initializer);

  return get_only(this->add_layer(layer, {input}, initializers));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::dense(
    parallel_tensor_guid_t const &input,
    nonnegative_int outDim,
    std::optional<Activation> activation,
    bool use_bias,
    DataType data_type,
    std::optional<InitializerAttrs> const &maybe_projection_initializer,
    std::optional<InitializerAttrs> const &maybe_bias_initializer,
    std::optional<std::string> const &maybe_name) {
  LinearAttrs attrs = LinearAttrs{
      /*out_channels=*/outDim,
      /*use_bias=*/use_bias,
      /*data_type=*/data_type,
      /*activation=*/activation,
      /*regularizer=*/std::nullopt,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::vector<InitializerAttrs> initializers =
      throw_if_unexpected(get_initializers(attrs,
                                           get_reduced_shape(input_shape),
                                           maybe_projection_initializer,
                                           maybe_bias_initializer));

  return get_only(this->add_layer(layer, {input}, initializers));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::embedding(
    parallel_tensor_guid_t const &input,
    nonnegative_int num_entries,
    nonnegative_int outDim,
    AggregateOp aggr,
    DataType dtype,
    std::optional<InitializerAttrs> const &maybe_kernel_initializer,
    std::optional<std::string> const &maybe_name) {

  EmbeddingAttrs attrs = EmbeddingAttrs{
      /*num_entries=*/num_entries,
      /*out_channels=*/outDim,
      /*aggr=*/aggr,
      /*data_type=*/dtype,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  std::vector<InitializerAttrs> initializers =
      get_initializers(attrs, maybe_kernel_initializer);

  return get_only(this->add_layer(layer, {input}, initializers));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::multihead_attention(
    parallel_tensor_guid_t const &query,
    parallel_tensor_guid_t const &key,
    parallel_tensor_guid_t const &value,
    nonnegative_int embed_dim,
    nonnegative_int num_heads,
    std::optional<nonnegative_int> maybe_kdim,
    std::optional<nonnegative_int> maybe_vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    std::optional<InitializerAttrs> maybe_weights_initializer,
    std::optional<InitializerAttrs> maybe_input_bias_initializer,
    std::optional<InitializerAttrs> maybe_output_bias_initializer,
    std::optional<std::string> const &maybe_name) {

  nonnegative_int kdim = maybe_kdim.value_or(embed_dim);
  nonnegative_int vdim = maybe_vdim.value_or(embed_dim);

  MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
      /*embed_dim=*/embed_dim,
      /*num_heads=*/num_heads,
      /*kdim=*/kdim,
      /*vdim=*/vdim,
      /*dropout=*/dropout,
      /*bias=*/bias,
      /*add_bias_kv=*/add_bias_kv,
      /*add_zero_attn=*/add_zero_attn,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  std::vector<InitializerAttrs> initializers = throw_if_unexpected(
      get_initializers(attrs,
                       get_reduced_shape(this->get_shape(query)),
                       get_reduced_shape(this->get_shape(key)),
                       get_reduced_shape(this->get_shape(value)),
                       maybe_weights_initializer,
                       maybe_input_bias_initializer,
                       maybe_output_bias_initializer));

  return get_only(this->add_layer(layer, {query, key, value}, initializers));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::batch_norm(
    parallel_tensor_guid_t const &input,
    bool affine,
    std::optional<Activation> const &activation,
    float eps,
    std::optional<float> const &momentum,
    std::optional<std::string> const &maybe_name) {

  if (activation.has_value() && activation.value() != Activation::RELU) {
    throw mk_runtime_error(fmt::format(
        "batch_norm currently only supports (1) no activation function, or (2) "
        "relu activation function, but received {}. "
        "If you need support for additional activation functions, please "
        "create an issue.",
        activation));
  }

  BatchNormAttrs attrs = BatchNormAttrs{
      /*relu=*/activation.has_value(),
      /*affine=*/affine,
      /*eps=*/eps,
      /*momentum=*/momentum,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::vector<ParallelTensorAttrs> weights;

  std::vector<InitializerAttrs> initializers =
      throw_if_unexpected(get_initializers(attrs));

  return get_only(this->add_layer(layer, {input}, initializers));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::element_unary(
    ElementUnaryAttrs const &attrs,
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {input}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::relu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      OperatorType::RELU,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::identity(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      OperatorType::IDENTITY,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::gelu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      OperatorType::GELU,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::sigmoid(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      OperatorType::SIGMOID,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::tanh(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      OperatorType::TANH,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::elu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      OperatorType::ELU,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_partition(
    parallel_tensor_guid_t const &input,
    ff_dim_t dim,
    nonnegative_int degree,
    std::optional<std::string> const &maybe_name) {

  RepartitionAttrs attrs = RepartitionAttrs{
      /*repartition_dim=*/dim,
      /*repartition_degree=*/degree,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {input}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_combine(
    parallel_tensor_guid_t const &input,
    ff_dim_t dim,
    nonnegative_int degree,
    std::optional<std::string> const &maybe_name) {

  CombineAttrs attrs = CombineAttrs{
      /*combine_dim=*/dim,
      /*combine_degree=*/degree,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {input}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_replicate(
    parallel_tensor_guid_t const &input,
    nonnegative_int degree,
    std::optional<std::string> const &maybe_name) {

  ReplicateAttrs attrs = ReplicateAttrs{degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {input}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_reduce(
    parallel_tensor_guid_t const &input,
    nonnegative_int degree,
    std::optional<std::string> const &maybe_name) {

  ReductionAttrs attrs = ReductionAttrs{degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return get_only(this->add_layer(layer, {input}, {}));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::as_type(
    parallel_tensor_guid_t const &input,
    DataType goal_datatype,
    std::string const &name) {
  DataType input_datatype = this->get_shape(input).data_type;
  if (input_datatype == goal_datatype) {
    return input;
  } else if (can_strictly_promote_datatype_from_to(input_datatype,
                                                   goal_datatype)) {
    return this->cast(input, goal_datatype, name);
  } else {
    throw mk_runtime_error(
        fmt::format("Could not convert provided tensor data type {} to "
                    "desired data type {}",
                    input_datatype,
                    goal_datatype));
  }
}

ParallelTensorShape ParallelComputationGraphBuilder::get_shape(
    parallel_tensor_guid_t const &t) const {
  return get_parallel_tensor_attrs(this->pcg, t).shape;
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add_weight(
    ParallelTensorShape const &par_weight_shape,
    InitializerAttrs const &initializer,
    std::optional<std::string> const &weight_name) {
  TensorShape unpar_weight_shape = get_reduced_shape(par_weight_shape);

  ParallelLayerAttrs weight_layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{WeightAttrs{
          /*shape=*/unpar_weight_shape,
          /*initializer=*/initializer,
      }},
      weight_name,
  };

  parallel_tensor_guid_t current_weight_tensor = get_only(
      add_parallel_layer(this->pcg, weight_layer_attrs, {}, {}).outputs);

  for (ParallelOpAttrs const &parallel_op_attr :
       generate_weight_transform(unpar_weight_shape, par_weight_shape)) {

    ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
        pcg_op_attrs_from_parallel_op_attrs(parallel_op_attr),
        std::nullopt,
    };
    current_weight_tensor = get_only(
        add_parallel_layer(this->pcg, layer_attrs, {current_weight_tensor}, {})
            .outputs);
  }

  return current_weight_tensor;
}

static void check_incoming_tensor_roles(ParallelLayerAttrs const &layer,
                                        int num_inputs,
                                        int num_weights) {
  std::vector<IncomingTensorRole> correct =
      get_incoming_tensor_roles(layer.op_attrs, num_inputs + num_weights);
  std::vector<IncomingTensorRole> current = concat_vectors(
      std::vector<IncomingTensorRole>(num_inputs, IncomingTensorRole::INPUT),
      std::vector<IncomingTensorRole>(num_weights, IncomingTensorRole::WEIGHT));

  if (correct != current) {
    throw mk_runtime_error(
        fmt::format("check_incoming_tensor_roles found deviation in incoming "
                    "tensors: expected {}, received {}",
                    correct,
                    current));
  }
}

std::vector<parallel_tensor_guid_t> ParallelComputationGraphBuilder::add_layer(
    ParallelLayerAttrs const &layer,
    std::vector<parallel_tensor_guid_t> const &inputs,
    std::vector<InitializerAttrs> const &weight_initializers) {

  int num_weights_provided =
      count(weight_initializers, [](std::optional<InitializerAttrs> const &i) {
        return i.has_value();
      });

  check_incoming_tensor_roles(layer, inputs.size(), num_weights_provided);

  std::vector<ParallelTensorShape> input_shapes =
      transform(inputs, [&](parallel_tensor_guid_t const &i) {
        return this->get_shape(i);
      });

  std::vector<ParallelTensorShape> weight_shapes =
      get_weight_shapes(layer.op_attrs, input_shapes);
  std::vector<parallel_tensor_guid_t> weight_tensors =
      zip_with(weight_shapes,
               weight_initializers,
               [&](ParallelTensorShape const &weight_shape,
                   InitializerAttrs const &initializer) {
                 return this->add_weight(weight_shape, initializer);
               });

  return add_parallel_layer(this->pcg, layer, inputs, weight_tensors, {})
      .outputs;
}

} // namespace FlexFlow
