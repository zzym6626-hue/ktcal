#pragma once

#include "fast_path.h"
#include "rknn_api.h" // supplied by the Rockchip RKNN Runtime SDK, never vendored

#include <mutex>

namespace ktcal {
class RknnBackend final : public Backend {
 public:
  // Requires an independently converted causal STNet RKNN model with a single
  // float-compatible [1,16,2,60,80] input and two logits outputs [80],[60].
  explicit RknnBackend(const char* model_path, float input_divisor = 1.f);
  ~RknnBackend() override;
  RknnBackend(const RknnBackend&) = delete;
  RknnBackend& operator=(const RknnBackend&) = delete;
  Point run(const Window& frames) override;
  std::future<Point> start(Window frames) override;
 private:
  rknn_context context_ = 0;
  float divisor_;
  std::mutex mutex_;
};
} // namespace ktcal
