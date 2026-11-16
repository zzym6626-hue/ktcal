#pragma once
#include "sequence_path.h"
#include "rknn_api.h" // external Rockchip SDK, not supplied by this repository

namespace ktcal {
class RknnSequenceBackend final : public SequenceBackend {
 public:
  explicit RknnSequenceBackend(const char* model_path);
  ~RknnSequenceBackend() override;
  RknnSequenceBackend(const RknnSequenceBackend&) = delete;
  RknnSequenceBackend& operator=(const RknnSequenceBackend&) = delete;
  // Raw counts. Normalization is embedded in the exported model, never repeated here.
  SequenceLogits run(const Record& frames) override;
 private:
  rknn_context context_ = 0;
};
} // namespace ktcal
