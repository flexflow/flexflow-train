#ifndef _FF_METRICS_H_
#define _FF_METRICS_H_

#include "op-attrs/ops/loss_functions/loss_functions.h"
#include "pcg/metric.dtg.h"
#include "utils/fmt.h"
#include <unordered_set>

namespace FlexFlow {

class MetricsAttrs {
public:
  MetricsAttrs() = delete;
  MetricsAttrs(LossFunction, std::unordered_set<Metric> const &);

public:
  LossFunction loss_type;
  bool measure_accuracy;
  bool measure_categorical_crossentropy;
  bool measure_sparse_categorical_crossentropy;
  bool measure_mean_squared_error;
  bool measure_root_mean_squared_error;
  bool measure_mean_absolute_error;
};

} // namespace FlexFlow

#endif
