#include "sequence_path.h"

#include <algorithm>
#include <cmath>
#include <istream>
#include <stdexcept>

namespace ktcal {
namespace {
void finite(Pixel value) {
  if (!std::isfinite(value[0]) || !std::isfinite(value[1]))
    throw std::invalid_argument("non-finite coordinate");
}
void validate_record(const Record& frames) {
  if (frames.size() != kSequence) throw std::invalid_argument("expected exactly 100 frames");
  for (const auto& frame : frames) for (float value : frame)
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite event input");
}
float soft_pixel(const float* logits, int length) {
  for (int i = 0; i < length; ++i)
    if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite sequence logits");
  // Temperature division and normalized-coordinate roundtrip use float32,
  // as in the published NumPy soft expectation decode.
  const float maximum = *std::max_element(logits, logits + length) / .65f;
  float weights[80], sum = 0, result = 0;
  for (int i = 0; i < length; ++i) {
    weights[i] = std::exp(logits[i] / .65f - maximum);
    sum += weights[i];
  }
  for (int i = 0; i < length; ++i) result += (weights[i] / sum) * i;
  return (result / length) * length;
}
double distance(Pixel a, Pixel b) { return std::hypot(a[0] - b[0], a[1] - b[1]); }
Point point(Pixel value) { return {static_cast<float>(value[0]), static_cast<float>(value[1])}; }
} // namespace

const std::array<View, 10>& sequence_views() {
  static const std::array<View, 10> views = {{{0,0,false}, {0,0,true},
      {-2,0,false}, {-2,0,true}, {2,0,false}, {2,0,true},
      {0,-2,false}, {0,-2,true}, {0,2,false}, {0,2,true}}};
  return views;
}
Frame transform_view(const Frame& frame, View view) {
  if (std::abs(view.dx) >= kWidth || std::abs(view.dy) >= kHeight)
    throw std::invalid_argument("invalid view shift");
  Frame output{};
  for (int c = 0; c < 2; ++c) for (int y = 0; y < kHeight; ++y)
    for (int x = 0; x < kWidth; ++x) {
      const int sx = (view.flip ? kWidth - 1 - x : x) - view.dx, sy = y - view.dy;
      if (sx >= 0 && sx < kWidth && sy >= 0 && sy < kHeight)
        output[c * kWidth * kHeight + y * kWidth + x] = frame[c * kWidth * kHeight + sy * kWidth + sx];
    }
  return output;
}
Coordinates decode_sequence(const SequenceLogits& logits) {
  if (logits.x.size() != kSequence * kWidth || logits.y.size() != kSequence * kHeight)
    throw std::invalid_argument("expected x[1,100,80], y[1,100,60] logits");
  Coordinates output{};
  for (std::size_t t = 0; t < kSequence; ++t)
    output[t] = {soft_pixel(logits.x.data() + t * kWidth, kWidth),
                 soft_pixel(logits.y.data() + t * kHeight, kHeight)};
  return output;
}
Coordinates median_views(const std::array<Coordinates, 10>& predictions) {
  Coordinates output{};
  for (std::size_t t = 0; t < kSequence; ++t) for (int axis = 0; axis < 2; ++axis) {
    std::array<float, 10> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      finite(predictions[i][t]);
      values[i] = static_cast<float>(predictions[i][t][axis]);
    }
    std::sort(values.begin(), values.end());
    output[t][axis] = (values[4] + values[5]) * .5f;
  }
  return output;
}
Coordinates smooth_ma7(const Coordinates& coordinates) {
  Coordinates normalized{}, output{};
  for (std::size_t t = 0; t < kSequence; ++t) {
    finite(coordinates[t]);
    normalized[t] = {static_cast<float>(coordinates[t][0]) / 80.f,
                     static_cast<float>(coordinates[t][1]) / 60.f};
  }
  for (int t = 0; t < static_cast<int>(kSequence); ++t) for (int axis = 0; axis < 2; ++axis) {
    double sum = 0;
    for (int offset = -3; offset <= 3; ++offset)
      sum += normalized[std::clamp(t + offset, 0, static_cast<int>(kSequence) - 1)][axis] / 7.;
    output[t][axis] = static_cast<float>(sum) * (axis == 0 ? 80.f : 60.f);
  }
  return output;
}
bool read_record(std::istream& input, Record& frames) {
  frames.resize(kSequence);
  const auto bytes = static_cast<std::streamsize>(sizeof(Frame) * kSequence);
  input.read(reinterpret_cast<char*>(frames.data()), bytes);
  if (input.gcount() == 0 && input.eof()) { frames.clear(); return false; }
  if (input.gcount() != bytes || input.bad()) throw std::runtime_error("incomplete 100-frame record");
  validate_record(frames);
  return true;
}
Coordinates predict_record(const Record& frames, SequenceBackend& backend) {
  validate_record(frames);
  std::array<Coordinates, 10> cache{};
  Record transformed(kSequence);
  for (std::size_t i = 0; i < sequence_views().size(); ++i) {
    const auto view = sequence_views()[i];
    for (std::size_t t = 0; t < kSequence; ++t) transformed[t] = transform_view(frames[t], view);
    cache[i] = decode_sequence(backend.run(transformed));
    for (auto& value : cache[i]) {
      float x = static_cast<float>(value[0]), y = static_cast<float>(value[1]);
      if (view.flip) x = 79.f - x;
      value = {x - view.dx, y - view.dy};
    }
  }
  return median_views(cache);
}

void SequenceKalman::reset(Pixel observation, std::optional<Pixel> previous) {
  finite(observation);
  if (previous) finite(*previous);
  state_ = {observation[0], observation[1], previous ? observation[0] - (*previous)[0] : 0.,
            previous ? observation[1] - (*previous)[1] : 0.};
  for (auto& row : covariance_) row.fill(0);
  for (int i = 0; i < 4; ++i) covariance_[i][i] = 10.;
}
Pixel SequenceKalman::predict() {
  state_[0] += state_[2]; state_[1] += state_[3];
  const double f[4][4] = {{1,0,1,0}, {0,1,0,1}, {0,0,1,0}, {0,0,0,1}};
  double temp[4][4]{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 4; ++k) temp[i][j] += f[i][k] * covariance_[k][j];
  std::array<std::array<double,4>,4> result{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) result[i][j] += temp[i][k] * f[j][k];
    if (i == j) result[i][j] += i < 2 ? .05 : .2;
  }
  covariance_ = result;
  return position();
}
Pixel SequenceKalman::update(Pixel observation) {
  finite(observation);
  const double s00 = covariance_[0][0] + 32, s01 = covariance_[0][1];
  const double s10 = covariance_[1][0], s11 = covariance_[1][1] + 32;
  const double det = s00 * s11 - s01 * s10;
  if (!(det > 0) || !std::isfinite(det)) throw std::runtime_error("invalid Kalman covariance");
  double gain[4][2];
  for (int i = 0; i < 4; ++i) {
    gain[i][0] = (covariance_[i][0] * s11 - covariance_[i][1] * s10) / det;
    gain[i][1] = (covariance_[i][1] * s00 - covariance_[i][0] * s01) / det;
  }
  const double ex = observation[0] - state_[0], ey = observation[1] - state_[1];
  for (int i = 0; i < 4; ++i) state_[i] += gain[i][0] * ex + gain[i][1] * ey;
  double a[4][4]{}, temp[4][4]{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
    a[i][j] = (i == j) - (j == 0 ? gain[i][0] : 0.) - (j == 1 ? gain[i][1] : 0.);
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 4; ++k) temp[i][j] += a[i][k] * covariance_[k][j];
  std::array<std::array<double,4>,4> result{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) result[i][j] += temp[i][k] * a[j][k];
    result[i][j] += 32 * (gain[i][0] * gain[j][0] + gain[i][1] * gain[j][1]);
  }
  covariance_ = result;
  return position();
}
GateOutput SequenceGate::seed(Pixel global) {
  kalman_.reset(global); stats_ = {}; stats_.tier3 = stats_.stnet_requests = 1;
  seeded_ = true; predicted_ = false;
  return {global, 3, true};
}
Pixel SequenceGate::predict() {
  if (!seeded_ || predicted_) throw std::logic_error("gate prediction out of order");
  previous_ = kalman_.position(); prediction_ = kalman_.predict(); predicted_ = true;
  return prediction_;
}
GateOutput SequenceGate::correct(MatchResult match, Pixel global) {
  if (!predicted_) throw std::logic_error("gate correction needs prediction");
  finite(global); predicted_ = false;
  const Pixel matched = {match.position.x, match.position.y};
  const double innovation = distance(prediction_, matched);
  const bool large = !std::isfinite(innovation) || !std::isfinite(match.confidence) ||
      !std::isfinite(match.psr) || match.confidence < .28f || match.psr < 2.55f ||
      match.border || innovation >= 5.;
  const int tier = large ? 3 : innovation >= 3. ? 2 : 1;
  Pixel result;
  if (large) { kalman_.reset(global, previous_); result = kalman_.position(); ++stats_.tier3; }
  else {
    result = kalman_.update(matched);
    if (tier == 1) { ++stats_.tier1; } // No STNet consultation or template refresh.
    else {
      ++stats_.tier2;
      if (distance(result, global) > 1.) {
        kalman_.reset(global, previous_); result = kalman_.position(); ++stats_.medium_rollbacks;
      } else {
        const auto refined = kalman_.update(global);
        result = {.75 * global[0] + .25 * refined[0], .75 * global[1] + .25 * refined[1]};
        ++stats_.medium_refinements; ++stats_.fusion_calls;
      }
    }
  }
  if (tier != 1) ++stats_.stnet_requests;
  if (match.border) ++stats_.border_peaks;
  return {result, tier, tier != 1};
}
SequenceTracker::SequenceTracker(MatcherWeights weights)
    : matcher_(std::move(weights), MatcherConfig{37,32,true}) {}
SequenceResult SequenceTracker::track(const Record& frames, const Coordinates& global) const {
  validate_record(frames);
  for (auto value : global) finite(value);
  SequenceResult result;
  SequenceGate gate;
  result.raw[0] = gate.seed(global[0]).position; result.tiers[0] = 3;
  auto patch = matcher_.template_patch(frames[0], point(global[0]));
  for (std::size_t t = 1; t < kSequence; ++t) {
    const auto prediction = gate.predict();
    const auto match = matcher_.match_patch(patch, frames[t], point(prediction));
    const auto output = gate.correct(match, global[t]);
    result.raw[t] = output.position; result.tiers[t] = output.tier;
    if (output.refresh_template) patch = matcher_.template_patch(frames[t], point(global[t]));
  }
  result.stats = gate.stats(); result.filtered = smooth_ma7(result.raw);
  return result;
}
} // namespace ktcal
