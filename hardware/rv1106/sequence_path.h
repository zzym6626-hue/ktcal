#pragma once

#include "fast_path.h"
#include <array>
#include <iosfwd>
#include <vector>

namespace ktcal {
constexpr std::size_t kSequence = 100;
using Record = std::vector<Frame>; // heap: a float32 record alone is 3.84 MB
using Pixel = std::array<double, 2>;
using Coordinates = std::array<Pixel, kSequence>;
struct View { int dx, dy; bool flip; };
const std::array<View, 10>& sequence_views();
Frame transform_view(const Frame& frame, View view);
struct SequenceLogits { std::vector<float> x, y; };
Coordinates decode_sequence(const SequenceLogits& logits);
Coordinates median_views(const std::array<Coordinates, 10>& predictions);
Coordinates smooth_ma7(const Coordinates& coordinates);
// EOF is valid only between complete records, never padded or truncated.
bool read_record(std::istream& input, Record& frames);

class SequenceBackend {
 public:
  virtual ~SequenceBackend() = default;
  virtual SequenceLogits run(const Record& frames) = 0;
};
Coordinates predict_record(const Record& frames, SequenceBackend& backend);

// Double-precision KF mirrors the released NumPy state and Joseph update.
class SequenceKalman {
 public:
  void reset(Pixel observation, std::optional<Pixel> previous = std::nullopt);
  Pixel predict();
  Pixel update(Pixel observation);
  Pixel position() const { return {state_[0], state_[1]}; }
 private:
  std::array<double, 4> state_{};
  std::array<std::array<double, 4>, 4> covariance_{};
};
struct SequenceStats {
  std::size_t tier1 = 0, tier2 = 0, tier3 = 0;
  std::size_t medium_refinements = 0, medium_rollbacks = 0;
  std::size_t fusion_calls = 0, small_fusion_calls = 0;
  std::size_t stnet_requests = 0, border_peaks = 0;
};
struct GateOutput { Pixel position; int tier; bool refresh_template; };
class SequenceGate {
 public:
  GateOutput seed(Pixel global);
  Pixel predict();
  GateOutput correct(MatchResult match, Pixel global);
  const SequenceStats& stats() const { return stats_; }
 private:
  SequenceKalman kalman_;
  Pixel previous_{}, prediction_{};
  SequenceStats stats_{};
  bool seeded_ = false, predicted_ = false;
};
struct SequenceResult {
  Coordinates raw{}, filtered{};
  std::array<int, kSequence> tiers{};
  SequenceStats stats;
};
class SequenceTracker {
 public:
  explicit SequenceTracker(MatcherWeights weights);
  SequenceResult track(const Record& frames, const Coordinates& global) const;
 private:
  Matcher matcher_;
};
} // namespace ktcal
