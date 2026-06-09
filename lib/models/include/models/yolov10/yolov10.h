/**
 * @file yolov10.h
 *
 * @brief YOLOv10 detection model
 */

#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_YOLOV10_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_YOLOV10_H

#include "models/yolov10/yolov10_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper types

/**
 * @brief Hold per-layer tensor / num_channels information.
 */
struct YOLOv10LayerChannelTensor {
  positive_int channels_;
  tensor_guid_t tensor_;
};

/**
 * @brief Hold detection outputs.
 */
struct YOLOv10DetectHeadOutputs {
  tensor_guid_t boxes;              // (B, 4*reg_max, sum_i(Hi*Wi))
  tensor_guid_t scores;             // (B, nc,        sum_i(Hi*Wi))
  std::vector<tensor_guid_t> feats; // passthrough features
};

// Helper functions to construct the YOLOv10 model

/**
 * @brief Get the default YOLOv10 config.
 *
 * @details The configs here refer to the example at
 * https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v10/yolov10x.yaml.
 */
YOLOv10Config get_default_yolov10_config();

bool is_yolov10_repeat_module(YOLOv10Module module_type);

YOLOv10LayerChannelTensor create_yolov10_concat_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    std::vector<int> const &input_tensor_index,
    nonnegative_int concat_dim);

YOLOv10LayerChannelTensor create_yolov10_upsample_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache);

YOLOv10LayerChannelTensor
    create_yolov10_conv_module(ComputationGraphBuilder &cgb,
                               tensor_guid_t const &input_tensor,
                               positive_int const &channel_in,
                               std::vector<int> const &conv_module_args);

YOLOv10LayerChannelTensor
    create_yolov10_scdown_module(ComputationGraphBuilder &cgb,
                                 tensor_guid_t const &input_tensor,
                                 positive_int const &channel_in,
                                 std::vector<int> const &scdown_module_args);

YOLOv10LayerChannelTensor
    create_yolov10_sppf_module(ComputationGraphBuilder &cgb,
                               tensor_guid_t const &input_tensor,
                               positive_int const &channel_in,
                               std::vector<int> const &sppf_module_args);

YOLOv10LayerChannelTensor
    create_yolov10_psa_module(ComputationGraphBuilder &cgb,
                              tensor_guid_t const &input_tensor,
                              positive_int const &channel_in,
                              std::vector<int> const &psa_module_args);

YOLOv10LayerChannelTensor create_yolov10_bottleneck_module(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &input_tensor,
    positive_int const &channel_in,
    std::vector<int> const &bottleneck_module_args);

YOLOv10LayerChannelTensor
    create_yolov10_c2f_module(ComputationGraphBuilder &cgb,
                              tensor_guid_t const &input_tensor,
                              positive_int const &channel_in,
                              std::vector<int> const &c2f_module_args);

YOLOv10LayerChannelTensor
    create_yolov10_cib_module(ComputationGraphBuilder &cgb,
                              tensor_guid_t const &input_tensor,
                              positive_int const &channel_in,
                              std::vector<int> const &cib_module_args);

YOLOv10LayerChannelTensor
    create_yolov10_c2fcib_module(ComputationGraphBuilder &cgb,
                                 tensor_guid_t const &input_tensor,
                                 positive_int const &channel_in,
                                 std::vector<int> const &c2fcib_module_args);

YOLOv10LayerChannelTensor
    create_yolov10_detect_box_head_one_level(ComputationGraphBuilder &cgb,
                                             tensor_guid_t const &feat,
                                             positive_int const &feat_channels,
                                             int c2,
                                             int reg_max);

YOLOv10LayerChannelTensor create_yolov10_v10detect_cls_head_one_level(
    ComputationGraphBuilder &cgb,
    tensor_guid_t const &feat,
    positive_int const &feat_channels,
    int c3,
    int nc);

YOLOv10DetectHeadOutputs create_yolov10_v10detect_forward(
    ComputationGraphBuilder &cgb,
    std::vector<tensor_guid_t> const &feats,
    std::vector<positive_int> const &feat_channels,
    int nc,
    int reg_max);

YOLOv10LayerChannelTensor create_yolov10_base_module_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    YOLOv10Module module_type,
    std::vector<int> const &input_tensor_index,
    positive_int const &num_module_repeats,
    std::vector<int> const &module_args);

YOLOv10LayerChannelTensor create_yolov10_detect_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    YOLOv10Config const &model_config,
    std::vector<int> const &input_tensor_index,
    std::vector<int> const &module_args);

tensor_guid_t create_yolov10_tensor(ComputationGraphBuilder &cgb,
                                    FFOrdered<positive_int> const &dims,
                                    DataType const &data_type);

YOLOv10LayerChannelTensor create_yolov10_layer(
    ComputationGraphBuilder &cgb,
    YOLOv10Config const &model_config,
    YOLOv10LayerConfig const &layer_config,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache);

/**
 * @brief Get the YOLOv10 computation graph.
 *
 * @param YOLOv10Config The config of YOLOv10 model.
 * @return ComputationGraph The computation graph of a YOLOv10 model.
 */
ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config);

} // namespace FlexFlow

#endif
