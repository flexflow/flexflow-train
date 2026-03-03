#include "models/yolov10/yolov10.h"
#include "models/yolov10/yolov10_config.dtg.h"
#include "models/yolov10/yolov10_module.dtg.h"
#include "op-attrs/relative_ff_dim_t.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "op-attrs/tensor_dims.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/repeat.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/positive_int/positive_int.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace FlexFlow {

namespace {

template <typename T, typename... Ts>
constexpr bool is_one_of(T value, Ts... values) {
  return ((value == values) || ...);
}

positive_int get_module_num_repeats(positive_int num_repeats_in_config,
                                    float model_scales_depth) {
  if (num_repeats_in_config > 1) {
    return positive_int(
        std::max(int(std::round(num_repeats_in_config.int_from_positive_int() *
                                model_scales_depth)),
                 1));
  }

  return num_repeats_in_config;
}

int make_divisible(int input, int divisor) {
  return ((input + divisor - 1) / divisor) * divisor;
}

nonnegative_int autopad_for_yolov10_conv(int kernel_size, int dilation) {
  int const k_eff =
      (dilation > 1) ? (dilation * (kernel_size - 1) + 1) : kernel_size;
  int const p = k_eff / 2;
  return nonnegative_int(p);
}

template <typename T>
T get_arg_or_default(std::vector<int> const &args, size_t idx, T default_val) {
  return (idx < args.size()) ? T(args[idx]) : default_val;
}

} // namespace

YOLOv10Config get_default_yolov10_config() {

  constexpr auto get_default_yolov10_layers =
      []() -> std::vector<YOLOv10LayerConfig> {
    std::vector<YOLOv10LayerConfig> layers{};

    // Add all layers of the default model
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::Conv,
        /*module_args=*/{64, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::Conv,
        /*module_args=*/{128, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/3_p,
        /*module_type=*/YOLOv10Module::C2f,
        /*module_args=*/{128, 1},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::Conv,
        /*module_args=*/{256, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/6_p,
        /*module_type=*/YOLOv10Module::C2f,
        /*module_args=*/{256, 1},
    });
    // layers.push_back(YOLOv10LayerConfig{
    //     /*input_tensor_index=*/{-1},
    //     /*num_module_repeats=*/1_p,
    //     /*module_type=*/YOLOv10Module::SCDown,
    //     /*module_args=*/{512, 3, 2},
    // });
    // layers.push_back(YOLOv10LayerConfig{
    //     /*input_tensor_index=*/{-1},
    //     /*num_module_repeats=*/6_p,
    //     /*module_type=*/YOLOv10Module::C2fCIB,
    //     /*module_args=*/{512, 1},
    // });
    // layers.push_back(YOLOv10LayerConfig{
    //     /*input_tensor_index=*/{-1},
    //     /*num_module_repeats=*/1_p,
    //     /*module_type=*/YOLOv10Module::SCDown,
    //     /*module_args=*/{1024, 3, 2},
    // });
    // layers.push_back(YOLOv10LayerConfig{
    //     /*input_tensor_index=*/{-1},
    //     /*num_module_repeats=*/3_p,
    //     /*module_type=*/YOLOv10Module::C2fCIB,
    //     /*module_args=*/{1024, 1},
    // });
    // layers.push_back(YOLOv10LayerConfig{
    //     /*input_tensor_index=*/{-1},
    //     /*num_module_repeats=*/1_p,
    //     /*module_type=*/YOLOv10Module::SPPF,
    //     /*module_args=*/{1024, 5},
    // });
    // layers.push_back(YOLOv10LayerConfig{
    //     /*input_tensor_index=*/{-1},
    //     /*num_module_repeats=*/1_p,
    //     /*module_type=*/YOLOv10Module::PSA,
    //     /*module_args=*/{1024},
    // });

    return layers;
  };

  return YOLOv10Config{
      /*num_input_channels=*/3_p,
      /*num_classes=*/80_p,
      /*model_scales=*/{1.0, 1.25, 512},
      /*model_layers=*/get_default_yolov10_layers(),
      /*batch_size=*/64_p,
  };
}

bool is_yolov10_repeat_module(YOLOv10Module module_type) {
  if (is_one_of(module_type, YOLOv10Module::C2f, YOLOv10Module::C2fCIB)) {
    return true;
  }
  return false;
}

YOLOv10LayerChannelTensor create_yolov10_concat_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    std::vector<int> const &input_tensor_index,
    nonnegative_int concat_dim) {

  std::vector<tensor_guid_t> tensors{};
  int channel_out = 0;

  for (int const idx : input_tensor_index) {
    tensors.push_back(layers_cache[idx].tensor_);
    channel_out += layers_cache[idx].channels_.int_from_positive_int();
  }

  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/tensors,
      /*axis=*/relative_ff_dim_t{concat_dim.unwrap_nonnegative()});

  return {positive_int(channel_out), cat_tensor};
}

YOLOv10LayerChannelTensor create_yolov10_upsample_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache) {

  // TODO: implement this when the upsample operator is available
  return {layers_cache.back().channels_,
          cgb.identity(layers_cache.back().tensor_)};
}

YOLOv10LayerChannelTensor
    create_yolov10_conv_module(ComputationGraphBuilder &cgb,
                               tensor_guid_t const &input_tensor,
                               positive_int const &channel_in,
                               std::vector<int> const &conv_module_args) {
  // Get conv parameters
  // clang-format off
  positive_int channel_out = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/1, /*default_val=*/channel_in);
  positive_int kernel_size = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/2, /*default_val=*/1_p);
  positive_int stride = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/3, /*default_val=*/1_p);
  positive_int groups = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/4, /*default_val=*/1_p);
  bool use_activation = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/5, /*default_val=*/true);
  positive_int dilation = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/6, /*default_val=*/1_p);
  nonnegative_int padding = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/7, /*default_val=*/autopad_for_yolov10_conv(
                                                                                         /*kernel_size=*/kernel_size.int_from_positive_int(),
                                                                                         /*dilation=*/dilation.int_from_positive_int()));
  // clang-format on

  // Create conv layer
  tensor_guid_t conv = cgb.conv2d(
      /*input=*/input_tensor,
      /*outChannels=*/channel_out,
      /*kernelH=*/kernel_size,
      /*kernelW=*/kernel_size,
      /*strideH=*/stride,
      /*strideW=*/stride,
      /*paddingH=*/padding,
      /*paddingW=*/padding,
      /*activation=*/std::nullopt,
      /*groups=*/groups,
      /*use_bias=*/false);

  // Add batch norm and activation
  // TODO: YOLOv10 uses SiLU
  tensor_guid_t out = cgb.batch_norm(
      /*input=*/conv,
      /*affine=*/true,
      /*activation=*/
      use_activation ? std::make_optional(Activation::RELU) : std::nullopt,
      /*eps=*/1e-5,
      /*momentum=*/0.1);

  return {
      .channels_ = channel_out,
      .tensor_ = out,
  };
}

YOLOv10LayerChannelTensor
    create_yolov10_scdown_module(ComputationGraphBuilder &cgb,
                                 tensor_guid_t const &input_tensor,
                                 positive_int const &channel_in,
                                 std::vector<int> const &scdown_module_args) {

  std::vector<int> conv1_module_args = scdown_module_args;
  conv1_module_args[2] = 1; // Change kernel size to 1
  conv1_module_args[3] = 1; // Change stride to 1

  std::vector<int> conv2_module_args = scdown_module_args;
  conv2_module_args.push_back(conv2_module_args[1]); // groups = channel_out
  conv2_module_args.push_back(0);                    // use_activation = false

  YOLOv10LayerChannelTensor conv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/conv1_module_args);

  YOLOv10LayerChannelTensor conv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/conv1.tensor_,
      /*channel_in=*/conv1.channels_,
      /*conv_module_args=*/conv2_module_args);

  return conv2;
}

// SPPF: Spatial Pyramid Pooling - Fast
YOLOv10LayerChannelTensor
    create_yolov10_sppf_module(ComputationGraphBuilder &cgb,
                               tensor_guid_t const &input_tensor,
                               positive_int const &channel_in,
                               std::vector<int> const &sppf_module_args) {

  // sppf_module_args = [c1, c2, k]
  int c1 = get_arg_or_default(
      /*args=*/sppf_module_args,
      /*idx=*/0,
      /*default_val=*/channel_in.int_from_positive_int());
  int c2 = get_arg_or_default(
      /*args=*/sppf_module_args,
      /*idx=*/1,
      /*default_val=*/channel_in.int_from_positive_int());
  int k = get_arg_or_default(/*args=*/sppf_module_args,
                             /*idx=*/2,
                             /*default_val=*/5);
  int n = get_arg_or_default(/*args=*/sppf_module_args,
                             /*idx=*/3,
                             /*default_val=*/3);

  int c_hidden = c1 / 2;

  // ------------------------------------------------------------
  // conv_module_args indices:
  //   [0]=channel_in, [1]=channel_out, [2]=kernel_size, [3]=stride,
  //   [4]=groups, [5]=use_activation, [6]=dilation, [7]=padding
  // ------------------------------------------------------------
  std::vector<int> cv1_module_args(/*count=*/6, /*value=*/0);
  cv1_module_args[0] = c1;
  cv1_module_args[1] = c_hidden;
  cv1_module_args[2] = 1;
  cv1_module_args[3] = 1;
  cv1_module_args[4] = 1;
  cv1_module_args[5] = 0; // use_activation = false

  YOLOv10LayerChannelTensor cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/cv1_module_args);

  // ------------------------------------------------------------
  // Sequential max pools: m(y[-1]) repeated n times
  // m = MaxPool2d(k, stride=1, padding=k//2)
  // ------------------------------------------------------------
  std::vector<tensor_guid_t> y_tensors;
  y_tensors.push_back(cv1.tensor_);

  tensor_guid_t pooled = cv1.tensor_;
  for (int i = 0; i < n; i++) {
    pooled = cgb.pool2d(
        /*input=*/pooled,
        /*kernelH=*/positive_int(k),
        /*kernelW=*/positive_int(k),
        /*strideH=*/positive_int(1),
        /*strideW=*/positive_int(1),
        /*paddingH=*/nonnegative_int(k / 2),
        /*paddingW=*/nonnegative_int(k / 2),
        /*type=*/PoolOp::MAX,
        /*activation=*/std::nullopt);

    y_tensors.push_back(pooled);
  }

  // ------------------------------------------------------------
  // torch.cat(y, dim=1)  (concat along channels)
  // ------------------------------------------------------------
  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/y_tensors,
      /*axis=*/relative_ff_dim_t{1});

  // ------------------------------------------------------------
  // cv2: Conv(c_hidden*(n+1), c2, 1, 1)
  // ------------------------------------------------------------
  positive_int cat_channels = positive_int(c_hidden * (n + 1));
  std::vector<int> cv2_module_args(/*count=*/4, /*value=*/0);
  cv2_module_args[0] = cat_channels.int_from_positive_int();
  cv2_module_args[1] = c2;
  cv2_module_args[2] = 1;
  cv2_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*channel_in=*/cat_channels,
      /*conv_module_args=*/cv2_module_args);

  return cv2;
}

// PSA: Position-Sensitive Attention
YOLOv10LayerChannelTensor
    create_yolov10_psa_module(ComputationGraphBuilder &cgb,
                              tensor_guid_t const &input_tensor,
                              positive_int const &channel_in,
                              std::vector<int> const &psa_module_args) {

  // psa_module_args = [c1, c2]
  int c1 = get_arg_or_default(
      /*args=*/psa_module_args,
      /*idx=*/0,
      /*default_val=*/channel_in.int_from_positive_int());
  int c2 = get_arg_or_default(
      /*args=*/psa_module_args,
      /*idx=*/1,
      /*default_val=*/channel_in.int_from_positive_int());
  float expansion_ratio = 0.5;

  int c = static_cast<int>(c1 * expansion_ratio);

  // ------------------------------------------------------------
  // conv_module_args indices:
  //   [0]=channel_in, [1]=channel_out, [2]=kernel_size, [3]=stride,
  //   [4]=groups, [5]=use_activation, [6]=dilation, [7]=padding
  // ------------------------------------------------------------
  std::vector<int> cv1_module_args(/*count=*/4, /*value=*/0);
  cv1_module_args[0] = c1;
  cv1_module_args[1] = 2 * c;
  cv1_module_args[2] = 1;
  cv1_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/cv1_module_args);

  // ------------------------------------------------------------
  // Split: (a, b) = cv1(x).split((c, c), dim=1)
  // ------------------------------------------------------------
  // TODO: use dense layer for now before split op is available
  // TODO: uncomment the code below when split op is supported.
  tensor_guid_t temp_split_output_1 = cgb.dense(cv1.tensor_, positive_int(c));
  tensor_guid_t temp_split_output_2 = cgb.dense(cv1.tensor_, positive_int(c));
  std::vector<tensor_guid_t> ab = {temp_split_output_1, temp_split_output_2};

  // std::vector<tensor_guid_t> ab = cgb.split(
  //     /*input=*/cv1.tensor_,
  //     /*split=*/
  //     std::vector<nonnegative_int>{nonnegative_int(c), nonnegative_int(c)},
  //     /*axis=*/relative_ff_dim_t{1});

  tensor_guid_t a_tensor = ab[0];
  tensor_guid_t b_tensor = ab[1];
  positive_int b_channels = positive_int(c);

  // ------------------------------------------------------------
  // b = b + attn(b)
  // ------------------------------------------------------------
  int num_heads_int = std::max(c / 64, 1);
  positive_int num_heads = positive_int(num_heads_int);

  // From Python Attention:
  //   head_dim = c // num_heads
  //   key_dim  = int(head_dim * 0.5)
  int head_dim = c / num_heads_int;
  int key_dim = static_cast<int>(static_cast<float>(head_dim) * 0.5f);

  tensor_guid_t attn_out = cgb.multihead_attention(
      /*query=*/b_tensor,
      /*key=*/b_tensor,
      /*value=*/b_tensor,
      /*embed_dim=*/b_channels,
      /*num_heads=*/num_heads,
      /*kdim=*/positive_int(key_dim),
      /*vdim=*/std::nullopt,
      /*dropout=*/0.0f,
      /*bias=*/false);

  tensor_guid_t b1 = cgb.add(/*x=*/b_tensor, /*y=*/attn_out);

  // ------------------------------------------------------------
  // FFN: Sequential(Conv(c, 2*c, 1), Conv(2*c, c, 1, act=False))
  // b = b + ffn(b)
  // ------------------------------------------------------------
  std::vector<int> ffn_cv1_args(/*count=*/4, /*value=*/0);
  ffn_cv1_args[0] = c;
  ffn_cv1_args[1] = 2 * c;
  ffn_cv1_args[2] = 1;
  ffn_cv1_args[3] = 1;

  YOLOv10LayerChannelTensor ffn1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/b1,
      /*channel_in=*/b_channels,
      /*conv_module_args=*/ffn_cv1_args);

  std::vector<int> ffn_cv2_args(/*count=*/6, /*value=*/0);
  ffn_cv2_args[0] = 2 * c;
  ffn_cv2_args[1] = c;
  ffn_cv2_args[2] = 1;
  ffn_cv2_args[3] = 1;
  ffn_cv2_args[4] = 1;
  ffn_cv2_args[5] = 0; // use_activation = false

  YOLOv10LayerChannelTensor ffn2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/ffn1.tensor_,
      /*channel_in=*/ffn1.channels_,
      /*conv_module_args=*/ffn_cv2_args);

  tensor_guid_t b2 = cgb.add(/*x=*/b1, /*y=*/ffn2.tensor_);

  // ------------------------------------------------------------
  // cat((a, b2), dim=1) then cv2: Conv(2*c, c1, 1, 1)
  // ------------------------------------------------------------
  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/std::vector<tensor_guid_t>{a_tensor, b2},
      /*axis=*/relative_ff_dim_t{1});

  positive_int cat_channels = positive_int(2 * c);

  std::vector<int> cv2_module_args(/*count=*/4, /*value=*/0);
  cv2_module_args[0] = cat_channels.int_from_positive_int();
  cv2_module_args[1] = c1;
  cv2_module_args[2] = 1;
  cv2_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*channel_in=*/cat_channels,
      /*conv_module_args=*/cv2_module_args);

  return cv2;
}

// Standard Bottleneck
YOLOv10LayerChannelTensor create_yolov10_bottleneck_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    positive_int const &channel_in,
    std::vector<int> const &bottleneck_module_args) {

  // bottleneck_module_args = [c1, c2, shortcut]
  int c1 = get_arg_or_default(
      /*args=*/bottleneck_module_args,
      /*idx=*/0,
      /*default_val=*/channel_in.int_from_positive_int());
  int c2 = get_arg_or_default(
      /*args=*/bottleneck_module_args,
      /*idx=*/1,
      /*default_val=*/channel_in.int_from_positive_int());
  bool shortcut = get_arg_or_default(
      /*args=*/bottleneck_module_args,
      /*idx=*/2,
      /*default_val=*/true);
  float expansion_ratio = get_arg_or_default(
      /*args=*/bottleneck_module_args,
      /*idx=*/3,
      /*default_val=*/0.5f);

  int c_hidden = static_cast<int>(static_cast<float>(c2) * expansion_ratio);

  // ------------------------------------------------------------
  // cv1: Conv(c1, c_hidden, 3, 1)
  // conv_module_args indices:
  //   [0]=channel_in, [1]=channel_out, [2]=kernel_size, [3]=stride,
  //   [4]=groups, [5]=use_activation, [6]=dilation, [7]=padding
  // ------------------------------------------------------------
  std::vector<int> cv1_module_args(/*count=*/4, /*value=*/0);
  cv1_module_args[0] = c1;
  cv1_module_args[1] = c_hidden;
  cv1_module_args[2] = 3;
  cv1_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/cv1_module_args);

  // ------------------------------------------------------------
  // cv2: Conv(c_hidden, c2, 3, 1)
  // ------------------------------------------------------------
  std::vector<int> cv2_module_args(/*count=*/4, /*value=*/0);
  cv2_module_args[0] = c_hidden;
  cv2_module_args[1] = c2;
  cv2_module_args[2] = 3;
  cv2_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cv1.tensor_,
      /*channel_in=*/cv1.channels_,
      /*conv_module_args=*/cv2_module_args);

  bool use_shortcut = shortcut && (c1 == c2);

  // clang-format off
  // if (use_shortcut) {
  //   // TODO: cgb.add doesn't work here for now because
  //   // TODO: input_tensor has shape: <TensorShape dims=<TensorDims ff_ordered=<ff_ordered [64, 160, 13, 80]>> data_type=FLOAT>
  //   // TODO: cv2.tensor_ has shape: <TensorShape dims=<TensorDims ff_ordered=<ff_ordered [64, 80, 13, 80]>> data_type=FLOAT>
  //   // TODO: Not sure if we need to support broadcast
  //   return {
  //       /*channels_=*/positive_int(c2),
  //       /*tensor_=*/cgb.add(/*lhs=*/input_tensor, /*rhs=*/cv2.tensor_),
  //   };
  // }
  // clang-format on

  return cv2;
}

// C2f: Faster Implementation of CSP Bottleneck with 2 convolutions
YOLOv10LayerChannelTensor
    create_yolov10_c2f_module(ComputationGraphBuilder &cgb,
                              tensor_guid_t const &input_tensor,
                              positive_int const &channel_in,
                              std::vector<int> const &c2f_module_args) {

  // c2f_module_args = [c1, c2, n, shortcut, g, e]
  int c1 = get_arg_or_default(
      /*args=*/c2f_module_args,
      /*idx=*/0,
      /*default_val=*/channel_in.int_from_positive_int());
  int c2 = get_arg_or_default(
      /*args=*/c2f_module_args,
      /*idx=*/1,
      /*default_val=*/channel_in.int_from_positive_int());
  int n = get_arg_or_default(
      /*args=*/c2f_module_args,
      /*idx=*/2,
      /*default_val=*/1);
  bool shortcut = get_arg_or_default(
      /*args=*/c2f_module_args,
      /*idx=*/3,
      /*default_val=*/false);
  int g = get_arg_or_default(
      /*args=*/c2f_module_args,
      /*idx=*/4,
      /*default_val=*/1);
  float e = get_arg_or_default(
      /*args=*/c2f_module_args,
      /*idx=*/5,
      /*default_val=*/0.5f);

  int c_hidden = static_cast<int>(static_cast<float>(c2) * e);

  // ------------------------------------------------------------
  // cv1: Conv(c1, 2*c_hidden, 1, 1)
  // conv_module_args indices:
  //   [0]=channel_in, [1]=channel_out, [2]=kernel_size, [3]=stride,
  //   [4]=groups, [5]=use_activation, [6]=dilation, [7]=padding
  // ------------------------------------------------------------
  std::vector<int> cv1_module_args(/*count=*/4, /*value=*/0);
  cv1_module_args[0] = c1;
  cv1_module_args[1] = 2 * c_hidden;
  cv1_module_args[2] = 1;
  cv1_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/cv1_module_args);

  // Split into (c_hidden, c_hidden) along channels (dim=1)
  // TODO: use dense layer for now before split op is available
  // TODO: uncomment the code below when split op is supported.
  tensor_guid_t temp_split_output_1 =
      cgb.dense(cv1.tensor_, positive_int(c_hidden));
  tensor_guid_t temp_split_output_2 =
      cgb.dense(cv1.tensor_, positive_int(c_hidden));
  std::vector<tensor_guid_t> y_split = {temp_split_output_1,
                                        temp_split_output_2};

  // std::vector<tensor_guid_t> y_split = cgb.split(
  //     /*input=*/cv1.tensor_,
  //     /*split=*/
  //     std::vector<nonnegative_int>{nonnegative_int(c_hidden),
  //                                  nonnegative_int(c_hidden)},
  //     /*axis=*/relative_ff_dim_t{1});

  // y = [y0, y1, ...]
  std::vector<tensor_guid_t> y_tensors;
  y_tensors.push_back(y_split[0]);
  y_tensors.push_back(y_split[1]);

  // ------------------------------------------------------------
  // m = ModuleList(Bottleneck(c, c, shortcut, g, e=1.0) for _ in range(n))
  // forward: y.extend(m(y[-1]) for m in self.m)
  // ------------------------------------------------------------
  std::vector<int> bottleneck_module_args;
  bottleneck_module_args.push_back(c_hidden);         // c1
  bottleneck_module_args.push_back(c_hidden);         // c2
  bottleneck_module_args.push_back(shortcut ? 1 : 0); // shortcut
  bottleneck_module_args.push_back(1);                // expansion_ratio = 1.0

  tensor_guid_t last = y_tensors.back();
  positive_int last_channels = positive_int(c_hidden);

  for (int i = 0; i < n; i++) {
    YOLOv10LayerChannelTensor bn = create_yolov10_bottleneck_module(
        /*cgb=*/cgb,
        /*input_tensor=*/last,
        /*channel_in=*/last_channels,
        /*bottleneck_module_args=*/bottleneck_module_args);

    last = bn.tensor_;
    last_channels = bn.channels_;
    y_tensors.push_back(last);
  }

  // ------------------------------------------------------------
  // cv2: Conv((2 + n) * c_hidden, c2, 1, 1)
  // ------------------------------------------------------------
  positive_int cat_channels = positive_int((2 + n) * c_hidden);

  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/y_tensors,
      /*axis=*/relative_ff_dim_t{1});

  std::vector<int> cv2_module_args(/*count=*/4, /*value=*/0);
  cv2_module_args[0] = cat_channels.int_from_positive_int();
  cv2_module_args[1] = c2;
  cv2_module_args[2] = 1;
  cv2_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*channel_in=*/cat_channels,
      /*conv_module_args=*/cv2_module_args);

  return cv2;
}

// CIB: Compact Inverted Block
YOLOv10LayerChannelTensor
    create_yolov10_cib_module(ComputationGraphBuilder &cgb,
                              tensor_guid_t const &input_tensor,
                              positive_int const &channel_in,
                              std::vector<int> const &cib_module_args) {

  // cib_module_args = [c1, c2, shortcut, e]
  int c1 = get_arg_or_default(
      /*args=*/cib_module_args,
      /*idx=*/0,
      /*default_val=*/channel_in.int_from_positive_int());
  int c2 = get_arg_or_default(
      /*args=*/cib_module_args,
      /*idx=*/1,
      /*default_val=*/channel_in.int_from_positive_int());
  bool shortcut = get_arg_or_default(
      /*args=*/cib_module_args,
      /*idx=*/2,
      /*default_val=*/true);
  float e = get_arg_or_default(
      /*args=*/cib_module_args,
      /*idx=*/3,
      /*default_val=*/0.5f);

  int c_hidden = static_cast<int>(static_cast<float>(c2) * e); // c_

  bool use_shortcut = shortcut && (c1 == c2);

  // ------------------------------------------------------------
  // cv1 = Sequential(
  //   1) Conv(c1, c1, 3, g=c1)
  //   2) Conv(c1, 2*c_hidden, 1)
  //   3) Conv(2*c_hidden, 2*c_hidden, 3, g=2*c_hidden)
  //   4) Conv(2*c_hidden, c2, 1)
  //   5) Conv(c2, c2, 3, g=c2)
  // ------------------------------------------------------------

  // 1) Conv(c1, c1, 3, stride=1, groups=c1)
  std::vector<int> conv1_args(/*count=*/5, /*value=*/0);
  conv1_args[0] = c1;
  conv1_args[1] = c1;
  conv1_args[2] = 3;
  conv1_args[3] = 1;
  conv1_args[4] = c1; // groups=c1

  YOLOv10LayerChannelTensor y1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/conv1_args);

  // 2) Conv(c1, 2*c_hidden, 1, 1)
  std::vector<int> conv2_args(/*count=*/4, /*value=*/0);
  conv2_args[0] = c1;
  conv2_args[1] = 2 * c_hidden;
  conv2_args[2] = 1;
  conv2_args[3] = 1;

  YOLOv10LayerChannelTensor y2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y1.tensor_,
      /*channel_in=*/y1.channels_,
      /*conv_module_args=*/conv2_args);

  // 3) Conv(2*c_hidden, 2*c_hidden, 3, stride=1, groups=2*c_hidden)
  std::vector<int> conv3_args(/*count=*/5, /*value=*/0);
  conv3_args[0] = 2 * c_hidden;
  conv3_args[1] = 2 * c_hidden;
  conv3_args[2] = 3;
  conv3_args[3] = 1;
  conv3_args[4] = 2 * c_hidden; // groups=2*c_hidden

  YOLOv10LayerChannelTensor y3 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y2.tensor_,
      /*channel_in=*/y2.channels_,
      /*conv_module_args=*/conv3_args);

  // 4) Conv(2*c_hidden, c2, 1, stride=1)
  std::vector<int> conv4_args(/*count=*/4, /*value=*/0);
  conv4_args[0] = 2 * c_hidden;
  conv4_args[1] = c2;
  conv4_args[2] = 1;
  conv4_args[3] = 1;

  YOLOv10LayerChannelTensor y4 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y3.tensor_,
      /*channel_in=*/y3.channels_,
      /*conv_module_args=*/conv4_args);

  // 5) Conv(c2, c2, 3, stride=1, groups=c2)
  std::vector<int> conv5_args(/*count=*/5, /*value=*/0);
  conv5_args[0] = c2;
  conv5_args[1] = c2;
  conv5_args[2] = 3;
  conv5_args[3] = 1;
  conv5_args[4] = c2; // groups=c2

  YOLOv10LayerChannelTensor y5 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y4.tensor_,
      /*channel_in=*/y4.channels_,
      /*conv_module_args=*/conv5_args);

  if (use_shortcut) {
    return {
        /*channels_=*/positive_int(c2),
        /*tensor_=*/cgb.add(/*lhs=*/input_tensor, /*rhs=*/y5.tensor_),
    };
  }

  return y5;
}

// C2fCIB: a convolutional block with C2f and CIB modules.
// This is a C2f variant where the repeated blocks are CIB instead of
// Bottleneck.
YOLOv10LayerChannelTensor
    create_yolov10_c2fcib_module(ComputationGraphBuilder &cgb,
                                 tensor_guid_t const &input_tensor,
                                 positive_int const &channel_in,
                                 std::vector<int> const &c2fcib_module_args) {

  // c2fcib_module_args = [c1, c2, n, shortcut, g, e]
  int c1 = get_arg_or_default(
      /*args=*/c2fcib_module_args,
      /*idx=*/0,
      /*default_val=*/channel_in.int_from_positive_int());
  int c2 = get_arg_or_default(
      /*args=*/c2fcib_module_args,
      /*idx=*/1,
      /*default_val=*/channel_in.int_from_positive_int());
  int n = get_arg_or_default(
      /*args=*/c2fcib_module_args,
      /*idx=*/2,
      /*default_val=*/1);
  bool shortcut = get_arg_or_default(
      /*args=*/c2fcib_module_args,
      /*idx=*/3,
      /*default_val=*/false);
  int g = get_arg_or_default(
      /*args=*/c2fcib_module_args,
      /*idx=*/4,
      /*default_val=*/1);
  float e = get_arg_or_default(
      /*args=*/c2fcib_module_args,
      /*idx=*/5,
      /*default_val=*/0.5f);

  int c_hidden = static_cast<int>(static_cast<float>(c2) * e);

  // ------------------------------------------------------------
  // cv1: Conv(c1, 2*c_hidden, 1, 1)
  // ------------------------------------------------------------
  std::vector<int> cv1_module_args(/*count=*/4, /*value=*/0);
  cv1_module_args[0] = c1;
  cv1_module_args[1] = 2 * c_hidden;
  cv1_module_args[2] = 1;
  cv1_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/cv1_module_args);

  // Split into (c_hidden, c_hidden) along channels (dim=1)
  // TODO: use dense layer for now before split op is available
  // TODO: uncomment the code below when split op is supported.
  tensor_guid_t temp_split_output_1 =
      cgb.dense(cv1.tensor_, positive_int(c_hidden));
  tensor_guid_t temp_split_output_2 =
      cgb.dense(cv1.tensor_, positive_int(c_hidden));
  std::vector<tensor_guid_t> y_split = {temp_split_output_1,
                                        temp_split_output_2};

  // std::vector<tensor_guid_t> y_split = cgb.split(
  //     /*input=*/cv1.tensor_,
  //     /*split=*/
  //     std::vector<nonnegative_int>{nonnegative_int(c_hidden),
  //                                  nonnegative_int(c_hidden)},
  //     /*axis=*/relative_ff_dim_t{1});

  std::vector<tensor_guid_t> y_tensors;
  y_tensors.push_back(y_split[0]);
  y_tensors.push_back(y_split[1]);

  // ------------------------------------------------------------
  // m = ModuleList(CIB(c_hidden, c_hidden, shortcut, e=1.0) for _ in range(n))
  // ------------------------------------------------------------
  std::vector<int> cib_module_args;
  cib_module_args.push_back(c_hidden);         // c1
  cib_module_args.push_back(c_hidden);         // c2
  cib_module_args.push_back(shortcut ? 1 : 0); // shortcut
  cib_module_args.push_back(1);                // e = 1.0

  tensor_guid_t last = y_tensors.back();
  positive_int last_channels = positive_int(c_hidden);

  for (int i = 0; i < n; i++) {
    YOLOv10LayerChannelTensor cib = create_yolov10_cib_module(
        /*cgb=*/cgb,
        /*input_tensor=*/last,
        /*channel_in=*/last_channels,
        /*cib_module_args=*/cib_module_args);

    last = cib.tensor_;
    last_channels = cib.channels_;
    y_tensors.push_back(last);
  }

  // ------------------------------------------------------------
  // cv2: Conv((2 + n) * c_hidden, c2, 1, 1)
  // ------------------------------------------------------------
  positive_int cat_channels = positive_int((2 + n) * c_hidden);

  tensor_guid_t cat_tensor = cgb.concat(
      /*tensors=*/y_tensors,
      /*axis=*/relative_ff_dim_t{1});

  std::vector<int> cv2_module_args(/*count=*/4, /*value=*/0);
  cv2_module_args[0] = cat_channels.int_from_positive_int();
  cv2_module_args[1] = c2;
  cv2_module_args[2] = 1;
  cv2_module_args[3] = 1;

  YOLOv10LayerChannelTensor cv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/cat_tensor,
      /*channel_in=*/cat_channels,
      /*conv_module_args=*/cv2_module_args);

  return cv2;
}

YOLOv10LayerChannelTensor
    create_yolov10_detect_box_head_one_level(ComputationGraphBuilder &cgb,
                                             tensor_guid_t const &feat,
                                             positive_int const &feat_channels,
                                             int c2,
                                             int reg_max) {
  std::vector<int> conv1_args(/*count=*/4, /*value=*/0);
  conv1_args[0] = feat_channels.int_from_positive_int();
  conv1_args[1] = c2;
  conv1_args[2] = 3;
  conv1_args[3] = 1;

  YOLOv10LayerChannelTensor y1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/feat,
      /*channel_in=*/feat_channels,
      /*conv_module_args=*/conv1_args);

  std::vector<int> conv2_args(/*count=*/4, /*value=*/0);
  conv2_args[0] = c2;
  conv2_args[1] = c2;
  conv2_args[2] = 3;
  conv2_args[3] = 1;

  YOLOv10LayerChannelTensor y2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y1.tensor_,
      /*channel_in=*/y1.channels_,
      /*conv_module_args=*/conv2_args);

  // nn.Conv2d(c2, 4*reg_max, 1) (no activation)
  std::vector<int> conv3_args(/*count=*/6, /*value=*/0);
  conv3_args[0] = c2;
  conv3_args[1] = 4 * reg_max;
  conv3_args[2] = 1;
  conv3_args[3] = 1;
  conv3_args[4] = 1;
  conv3_args[5] = 0; // use_activation=false

  YOLOv10LayerChannelTensor y3 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/y2.tensor_,
      /*channel_in=*/y2.channels_,
      /*conv_module_args=*/conv3_args);

  return y3;
}

YOLOv10LayerChannelTensor create_yolov10_v10detect_cls_head_one_level(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &feat,
    positive_int const &feat_channels,
    int c3,
    int nc) {
  int x = feat_channels.int_from_positive_int();

  // (Conv(x,x,3,g=x) -> Conv(x,c3,1))
  std::vector<int> b1_conv1_args(/*count=*/5, /*value=*/0);
  b1_conv1_args[0] = x;
  b1_conv1_args[1] = x;
  b1_conv1_args[2] = 3;
  b1_conv1_args[3] = 1;
  b1_conv1_args[4] = x;

  YOLOv10LayerChannelTensor b1_1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/feat,
      /*channel_in=*/feat_channels,
      /*conv_module_args=*/b1_conv1_args);

  std::vector<int> b1_conv2_args(/*count=*/4, /*value=*/0);
  b1_conv2_args[0] = x;
  b1_conv2_args[1] = c3;
  b1_conv2_args[2] = 1;
  b1_conv2_args[3] = 1;

  YOLOv10LayerChannelTensor b1_2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/b1_1.tensor_,
      /*channel_in=*/b1_1.channels_,
      /*conv_module_args=*/b1_conv2_args);

  // (Conv(c3,c3,3,g=c3) -> Conv(c3,c3,1))
  std::vector<int> b2_conv1_args(/*count=*/5, /*value=*/0);
  b2_conv1_args[0] = c3;
  b2_conv1_args[1] = c3;
  b2_conv1_args[2] = 3;
  b2_conv1_args[3] = 1;
  b2_conv1_args[4] = c3;

  YOLOv10LayerChannelTensor b2_1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/b1_2.tensor_,
      /*channel_in=*/b1_2.channels_,
      /*conv_module_args=*/b2_conv1_args);

  std::vector<int> b2_conv2_args(/*count=*/4, /*value=*/0);
  b2_conv2_args[0] = c3;
  b2_conv2_args[1] = c3;
  b2_conv2_args[2] = 1;
  b2_conv2_args[3] = 1;

  YOLOv10LayerChannelTensor b2_2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/b2_1.tensor_,
      /*channel_in=*/b2_1.channels_,
      /*conv_module_args=*/b2_conv2_args);

  // nn.Conv2d(c3, nc, 1) (no activation)
  std::vector<int> b3_args(/*count=*/6, /*value=*/0);
  b3_args[0] = c3;
  b3_args[1] = nc;
  b3_args[2] = 1;
  b3_args[3] = 1;
  b3_args[4] = 1;
  b3_args[5] = 0; // use_activation=false

  YOLOv10LayerChannelTensor out = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/b2_2.tensor_,
      /*channel_in=*/b2_2.channels_,
      /*conv_module_args=*/b3_args);

  return out;
}

YOLOv10DetectHeadOutputs create_yolov10_v10detect_forward(
    ComputationGraphBuilder &cgb,
    std::vector<tensor_guid_t> const &feats,
    std::vector<positive_int> const &feat_channels,
    int nc,
    int reg_max) {

  int nl = static_cast<int>(feats.size());

  int ch0 =
      feat_channels.empty() ? 0 : feat_channels[0].int_from_positive_int();
  int c2 = std::max(std::max(16, ch0 / 4), reg_max * 4);
  int c3 = std::max(ch0, std::min(nc, 100));

  std::vector<tensor_guid_t> box_views;
  std::vector<tensor_guid_t> cls_views;

  for (int i = 0; i < nl; i++) {
    YOLOv10LayerChannelTensor box_logits =
        create_yolov10_detect_box_head_one_level(
            /*cgb=*/cgb,
            /*feat=*/feats[i],
            /*feat_channels=*/feat_channels[i],
            /*c2=*/c2,
            /*reg_max=*/reg_max);

    YOLOv10LayerChannelTensor cls_logits =
        create_yolov10_v10detect_cls_head_one_level(
            /*cgb=*/cgb,
            /*feat=*/feats[i],
            /*feat_channels=*/feat_channels[i],
            /*c3=*/c3,
            /*nc=*/nc);

    // Query BCHW shape from the logits (or feats[i]; should be same H/W).
    TensorDims shape = cgb.get_shape(box_logits.tensor_).dims;

    nonnegative_int B =
        nonnegative_int(dim_at_idx(shape, relative_ff_dim_t(0)));
    nonnegative_int H =
        nonnegative_int(dim_at_idx(shape, relative_ff_dim_t(2)));
    nonnegative_int W =
        nonnegative_int(dim_at_idx(shape, relative_ff_dim_t(3)));
    nonnegative_int N =
        nonnegative_int(H.unwrap_nonnegative() * W.unwrap_nonnegative());

    // BCHW -> (B, C, H*W)
    // TODO: enable below after reshape operator is supported
    // tensor_guid_t box_view = cgb.reshape(
    //     /*input=*/box_logits.tensor_,
    //     /*shape=*/std::vector<nonnegative_int>{B, nonnegative_int(4 *
    //     reg_max),
    //                                            N});
    // tensor_guid_t cls_view = cgb.reshape(
    //     /*input=*/cls_logits.tensor_,
    //     /*shape=*/std::vector<nonnegative_int>{B, nonnegative_int(nc), N});

    // box_views.push_back(box_view);
    // cls_views.push_back(cls_view);
  }

  // Concat along token dim N (axis=2 for (B,C,N))
  tensor_guid_t boxes = cgb.concat(
      /*tensors=*/box_views,
      /*axis=*/relative_ff_dim_t{2});

  tensor_guid_t scores = cgb.concat(
      /*tensors=*/cls_views,
      /*axis=*/relative_ff_dim_t{2});

  return YOLOv10DetectHeadOutputs{
      /*boxes=*/boxes,
      /*scores=*/scores,
      /*feats=*/feats,
  };
}

YOLOv10LayerChannelTensor create_yolov10_base_module_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    YOLOv10Module module_type,
    std::vector<int> const &input_tensor_index,
    positive_int const &num_module_repeats,
    std::vector<int> const &module_args) {

  if (module_type == YOLOv10Module::Conv) {
    return create_yolov10_conv_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  if (module_type == YOLOv10Module::SCDown) {
    return create_yolov10_scdown_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  if (module_type == YOLOv10Module::SPPF) {
    return create_yolov10_sppf_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  if (module_type == YOLOv10Module::PSA) {
    return create_yolov10_psa_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  if (module_type == YOLOv10Module::C2f) {
    return create_yolov10_c2f_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  if (module_type == YOLOv10Module::C2fCIB) {
    return create_yolov10_c2fcib_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  // Shouldn't reach here
  return {layers_cache.back().channels_,
          cgb.identity(layers_cache.back().tensor_)};
}

tensor_guid_t create_yolov10_tensor(ComputationGraphBuilder &cgb,
                                    FFOrdered<positive_int> const &dims,
                                    DataType const &data_type) {
  TensorShape input_shape = TensorShape{
      TensorDims{dims},
      data_type,
  };
  return cgb.create_input(input_shape, CreateGrad::YES);
};

YOLOv10LayerChannelTensor create_yolov10_detect_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    YOLOv10Config const &model_config,
    std::vector<int> const &input_tensor_index,
    std::vector<int> const &module_args) {

  std::vector<tensor_guid_t> feats{};
  std::vector<positive_int> feat_channels{};
  for (int const idx : input_tensor_index) {
    feats.push_back(layers_cache[idx].tensor_);
    feat_channels.push_back(layers_cache[idx].channels_);
  }

  YOLOv10DetectHeadOutputs outputs = create_yolov10_v10detect_forward(
      /*cgb=*/cgb,
      /*feats=*/feats,
      /*feat_channels=*/feat_channels,
      /*nc=*/model_config.num_classes.int_from_positive_int(),
      /*reg_max=*/16);

  return {model_config.num_classes, outputs.boxes};
}

YOLOv10LayerChannelTensor create_yolov10_layer(
    ComputationGraphBuilder &cgb,
    YOLOv10Config const &model_config,
    YOLOv10LayerConfig const &layer_config,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache) {

  if (layer_config.module_type == YOLOv10Module::Concat) {
    return create_yolov10_concat_layer(
        cgb,
        layers_cache,
        /*input_tensor_index=*/layer_config.input_tensor_index,
        /*concat_dim=*/nonnegative_int(layer_config.module_args[0]));
  }

  if (layer_config.module_type == YOLOv10Module::Upsample) {
    return create_yolov10_upsample_layer(cgb, layers_cache);
  }

  if (layer_config.module_type == YOLOv10Module::v10Detect) {
    // Enrich module arguments
    std::vector<int> module_args = layer_config.module_args;
    for (int const idx : layer_config.input_tensor_index) {
      module_args.push_back(
          layers_cache[idx].channels_.int_from_positive_int());
    }

    return create_yolov10_detect_layer(
        cgb,
        layers_cache,
        /*model_config=*/model_config,
        /*input_tensor_index=*/layer_config.input_tensor_index,
        /*module_args=*/module_args);
  }

  // Handle other base modules below

  float model_scales_depth = model_config.model_scales.at(0);
  float model_scales_width = model_config.model_scales.at(1);
  int model_scales_max_channels = model_config.model_scales.at(2);

  positive_int num_module_repeats = get_module_num_repeats(
      layer_config.num_module_repeats, model_scales_depth);

  // Get number of input and output channels
  int input_tensor_index = layer_config.input_tensor_index.at(0);
  if (input_tensor_index == -1) {
    input_tensor_index = layers_cache.size() - 1;
  }

  int const channel_in =
      layers_cache.at(input_tensor_index).channels_.int_from_positive_int();

  int channel_out = layer_config.module_args.at(0);
  if (channel_out != model_config.num_classes) {
    // Scale the output channel size if needed
    channel_out = make_divisible(
        std::min(channel_out, model_scales_max_channels) * model_scales_width,
        8);
  }

  // Prepare module args
  std::vector<int> module_args{channel_in, channel_out};
  module_args.insert(module_args.end(),
                     layer_config.module_args.begin() + 1,
                     layer_config.module_args.end());

  if (is_yolov10_repeat_module(layer_config.module_type)) {
    // "Repeat" modules take the number of repeats as one of its arguments
    module_args.insert(module_args.begin() + 2,
                       num_module_repeats.int_from_positive_int());
    num_module_repeats = 1_p;
  }

  return create_yolov10_base_module_layer(
      /*cgb=*/cgb,
      /*layers_cache=*/layers_cache,
      /*module_type=*/layer_config.module_type,
      /*input_tensor_index=*/layer_config.input_tensor_index,
      /*num_module_repeats=*/num_module_repeats,
      /*module_args=*/module_args);
}

ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config) {

  ComputationGraphBuilder cgb;

  // Create the initial input tensor
  tensor_guid_t input = create_yolov10_tensor(
      cgb,
      FFOrdered{config.batch_size, config.num_input_channels, 50_p, 50_p},
      DataType::FLOAT);

  // Cache holding layer-wise information
  std::vector<YOLOv10LayerChannelTensor> layers_cache{YOLOv10LayerChannelTensor{
      .channels_ = config.num_input_channels,
      .tensor_ = input,
  }};

  for (size_t i = 0; i < config.model_layers.size(); i++) {
    const YOLOv10LayerConfig layer_config = config.model_layers[i];
    const YOLOv10LayerChannelTensor layer =
        create_yolov10_layer(cgb, config, layer_config, layers_cache);

    if (i == 0) {
      layers_cache.clear();
    }

    layers_cache.push_back(layer);
  }

  return cgb.computation_graph;
}

} // namespace FlexFlow
