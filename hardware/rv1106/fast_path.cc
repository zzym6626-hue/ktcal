#include "fast_path.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ktcal {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "matcher file requires IEEE754 32-bit float");
namespace {
constexpr int kTemplateWidth = 30, kTemplateHeight = 25;

float distance(Point a, Point b) { return std::hypot(a.x - b.x, a.y - b.y); }

float sample(const Frame& image, int channel, int x, int y) {
  return (x >= 0 && x < kWidth && y >= 0 && y < kHeight)
             ? image[channel * kWidth * kHeight + y * kWidth + x] : 0;
}

std::vector<float> crop(const Frame& image, Point center, int width, int height) {
  std::vector<float> result(2 * width * height);
  for (int c = 0; c < 2; ++c) {
    for (int y = 0; y < height; ++y) {
      const float fy = center.y + y - (height - 1) * 0.5f;
      const int y0 = static_cast<int>(std::floor(fy));
      const float wy = fy - y0;
      for (int x = 0; x < width; ++x) {
        const float fx = center.x + x - (width - 1) * 0.5f;
        const int x0 = static_cast<int>(std::floor(fx));
        const float wx = fx - x0;
        result[c * width * height + y * width + x] =
            (1 - wy) * ((1 - wx) * sample(image, c, x0, y0)
                        + wx * sample(image, c, x0 + 1, y0))
            + wy * ((1 - wx) * sample(image, c, x0, y0 + 1)
                    + wx * sample(image, c, x0 + 1, y0 + 1));
      }
    }
  }
  return result;
}

std::vector<float> features(const std::vector<float>& patch, int width, int height,
                            const MatcherWeights& weights) {
  const int size = width * height;
  std::vector<float> depth(2 * size, 0), result(weights.channels * size, 0);
  for (int c = 0; c < 2; ++c) {
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      float sum = 0;
      for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const int yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < height && xx >= 0 && xx < width)
          sum += patch[c * size + yy * width + xx]
                 * weights.depthwise[c * 9 + (dy + 1) * 3 + dx + 1];
      }
      depth[c * size + y * width + x] = sum;
    }
  }
  for (std::size_t c = 0; c < weights.channels; ++c)
    for (int i = 0; i < size; ++i)
      result[c * size + i] = std::max(0.f, weights.pointwise[2*c] * depth[i]
                                     + weights.pointwise[2*c+1] * depth[size+i]);
  return result;
}

void read_exact(std::ifstream& stream, char* destination, std::size_t bytes) {
  if (!stream.read(destination, static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("truncated matcher weight file");
}
} // namespace

Route route(float error_px, float confidence, bool border, float confidence_threshold) {
  if (!std::isfinite(confidence_threshold) || confidence_threshold < 0 ||
      confidence_threshold > 1)
    throw std::invalid_argument("invalid confidence threshold");
  if (!std::isfinite(error_px) || !std::isfinite(confidence) || border ||
      confidence < confidence_threshold)
    return Route::Reset;
  return error_px < 3 ? Route::Fast : error_px < 5 ? Route::Deferred : Route::Reset;
}

MatcherWeights load_matcher_weights(const char* file) {
  std::ifstream input(file, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open matcher weights");
  char magic[8];
  read_exact(input, magic, 8);
  if (std::string(magic, 8) != std::string("KTMATCH1", 8))
    throw std::runtime_error("invalid matcher weight format");
  MatcherWeights weights;
  read_exact(input, reinterpret_cast<char*>(&weights.channels), sizeof(weights.channels));
  if (weights.channels == 0 || weights.channels > 256)
    throw std::runtime_error("invalid matcher channel count");
  weights.pointwise.resize(2 * weights.channels);
  read_exact(input, reinterpret_cast<char*>(weights.depthwise.data()), 18 * sizeof(float));
  read_exact(input, reinterpret_cast<char*>(weights.pointwise.data()),
             weights.pointwise.size() * sizeof(float));
  if (input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("extra bytes in matcher weight file");
  for (float value : weights.depthwise)
    if (!std::isfinite(value)) throw std::runtime_error("invalid matcher weight");
  for (float value : weights.pointwise)
    if (!std::isfinite(value)) throw std::runtime_error("invalid matcher weight");
  return weights;
}

Matcher::Matcher(MatcherWeights weights, MatcherConfig config)
    : weights_(std::move(weights)), config_(config) {
  if (!weights_.channels || weights_.pointwise.size() != 2 * weights_.channels)
    throw std::invalid_argument("matcher requires depthwise and pointwise weights");
  if (config.search_width < kTemplateWidth || config.search_height < kTemplateHeight ||
      config.search_width > kWidth || config.search_height > kHeight)
    throw std::invalid_argument("invalid matcher search size");
}

std::vector<float> Matcher::template_patch(const Frame& image, Point center) const {
  return crop(image, center, kTemplateWidth, kTemplateHeight);
}

MatchResult Matcher::match(const Frame& previous, const Frame& current,
                           Point template_center, Point search_center) const {
  return match_patch(template_patch(previous, template_center), current, search_center);
}

MatchResult Matcher::match_patch(const std::vector<float>& patch, const Frame& current,
                                 Point search_center) const {
  if (patch.size() != 2 * kTemplateWidth * kTemplateHeight)
    throw std::invalid_argument("invalid template patch");
  const auto templ = features(patch, kTemplateWidth, kTemplateHeight, weights_);
  const int tw = kTemplateWidth, th = kTemplateHeight;
  const int sw = config_.search_width, sh = config_.search_height;
  const auto search = features(crop(current, search_center, sw, sh), sw, sh, weights_);
  const int rw = sw - tw + 1, rh = sh - th + 1;
  std::vector<float> response(rw * rh);
  float best = -std::numeric_limits<float>::infinity();
  int best_x = 0, best_y = 0;
  for (int y = 0; y <= sh - th; ++y) for (int x = 0; x <= sw - tw; ++x) {
    double dot = 0, norm_t = 0, norm_s = 0;
    for (std::size_t c = 0; c < weights_.channels; ++c)
      for (int row = 0; row < th; ++row) for (int col = 0; col < tw; ++col) {
        const float a = templ[c*tw*th + row*tw + col];
        const float b = search[c*sw*sh + (row+y)*sw + col+x];
        dot += static_cast<double>(a)*b;
        norm_t += static_cast<double>(a)*a;
        norm_s += static_cast<double>(b)*b;
      }
    const float value = static_cast<float>(dot / std::sqrt(std::max(1e-12, norm_t*norm_s)));
    response[y * rw + x] = value;
    if (value > best) { best = value; best_x = x; best_y = y; }
  }
  const bool border = best_x == 0 || best_x == rw - 1 || best_y == 0 || best_y == rh - 1;
  auto offset = [](float lower, float peak, float upper) {
    const float denominator = lower - 2 * peak + upper;
    return denominator < -1e-9f ? std::clamp(.5f * (lower - upper) / denominator, -.5f, .5f) : 0.f;
  };
  float dx = 0, dy = 0;
  if (config_.subpixel && best_x > 0 && best_x < rw - 1)
    dx = offset(response[best_y * rw + best_x - 1], best, response[best_y * rw + best_x + 1]);
  if (config_.subpixel && best_y > 0 && best_y < rh - 1)
    dy = offset(response[(best_y - 1) * rw + best_x], best, response[(best_y + 1) * rw + best_x]);
  double sum = 0, variance = 0;
  const int index = best_y * rw + best_x;
  for (int i = 0; i < rw * rh; ++i) if (i != index) sum += response[i];
  const double count = std::max(1, rw * rh - 1), mean = sum / count;
  for (int i = 0; i < rw * rh; ++i) if (i != index) variance += std::pow(response[i] - mean, 2);
  const float psr = static_cast<float>((best - mean) / std::sqrt(std::max(1e-12, variance / count)));
  // Match Python's normalized local-coordinate -> pixel roundtrip in float32.
  const float span_x = sw - tw, span_y = sh - th;
  return {{search_center.x + ((best_x + dx) / std::max(span_x, 1.f) - .5f) * span_x,
           search_center.y + ((best_y + dy) / std::max(span_y, 1.f) - .5f) * span_y},
          best, border, psr};
}

void Kalman::reset(Point observation, std::optional<Point> previous) {
  state_ = {observation.x, observation.y,
            previous ? observation.x - previous->x : 0,
            previous ? observation.y - previous->y : 0};
  for (auto& row : covariance_) row.fill(0);
  for (int i = 0; i < 4; ++i) covariance_[i][i] = 10;
}

Point Kalman::position() const {
  return {static_cast<float>(state_[0]), static_cast<float>(state_[1])};
}

Point Kalman::predict() {
  state_[0] += state_[2]; state_[1] += state_[3];
  const double f[4][4] = {{1,0,1,0},{0,1,0,1},{0,0,1,0},{0,0,0,1}};
  double temp[4][4]{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 4; ++k) temp[i][j] += f[i][k] * covariance_[k][j];
  std::array<std::array<double, 4>, 4> result{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) result[i][j] += temp[i][k] * f[j][k];
    if (i == j) result[i][j] += i < 2 ? 0.1 : 0.5;
  }
  covariance_ = result;
  return position();
}

Point Kalman::update(Point observation) {
  const double s00 = covariance_[0][0] + 2, s01 = covariance_[0][1];
  const double s10 = covariance_[1][0], s11 = covariance_[1][1] + 2;
  const double det = s00*s11 - s01*s10;
  if (!(det > 0)) throw std::runtime_error("singular Kalman covariance");
  double gain[4][2];
  for (int i = 0; i < 4; ++i) {
    gain[i][0] = (covariance_[i][0]*s11 - covariance_[i][1]*s10)/det;
    gain[i][1] = (covariance_[i][1]*s00 - covariance_[i][0]*s01)/det;
  }
  const double ex = observation.x - state_[0], ey = observation.y - state_[1];
  for (int i = 0; i < 4; ++i) state_[i] += gain[i][0]*ex + gain[i][1]*ey;
  double a[4][4]{}, temp[4][4]{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
    a[i][j] = (i == j) - (j == 0 ? gain[i][0] : 0) - (j == 1 ? gain[i][1] : 0);
  }
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 4; ++k) temp[i][j] += a[i][k]*covariance_[k][j];
  std::array<std::array<double, 4>, 4> result{};
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) result[i][j] += temp[i][k]*a[j][k];
    result[i][j] += 2 * (gain[i][0]*gain[j][0] + gain[i][1]*gain[j][1]);
  }
  covariance_ = result;
  return position();
}

Tracker::Tracker(Matcher matcher, float confidence_threshold)
    : matcher_(std::move(matcher)), confidence_threshold_(confidence_threshold) {
  if (!std::isfinite(confidence_threshold) || confidence_threshold < 0 ||
      confidence_threshold > 1)
    throw std::invalid_argument("invalid confidence threshold");
}

Window Tracker::history_window() const {
  Window window;
  const std::size_t padding = kHistory - history_.size();
  for (std::size_t i = 0; i < padding; ++i) window[i] = history_.front();
  for (std::size_t i = 0; i < history_.size(); ++i) window[i + padding] = history_[i];
  return window;
}

void Tracker::collect() {
  if (!pending_ || pending_->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    return;
  const auto frame = pending_->frame;
  const auto predicted = pending_->predicted;
  const auto previous = pending_->previous;
  const bool stale = frame < anchor_ || frame + 1 != frame_index_;
  // A superseded NPU failure must not poison a successful synchronous anchor.
  if (stale) {
    try { pending_->future.get(); } catch (...) { /* superseded */ }
    pending_.reset();
    ++async_discarded_;
    return;
  }
  const Point result = pending_->future.get();
  pending_.reset();
  // A late result cannot safely be applied to the current KF without history replay.
  if (!std::isfinite(result.x) || !std::isfinite(result.y)) { ++async_discarded_; return; }
  if (distance(result, predicted) > 1) kalman_.reset(result, previous);
  else kalman_.update(result);
  ++async_applied_;
  // The prior frame is the newest available template when the result lands.
  template_frame_ = history_.back();
  template_center_ = result;
}

void Tracker::finish() {
  if (pending_) {
    pending_->future.wait();
    collect();
  }
}

Output Tracker::tick(const Frame& frame, Backend& backend) {
  return process(frame, backend, std::nullopt);
}
Output Tracker::tick_with_match(const Frame& frame, Backend& backend, MatchResult match) {
  return process(frame, backend, match);
}

Output Tracker::process(const Frame& frame, Backend& backend,
                        std::optional<MatchResult> override_match) {
  collect();
  history_.push_back(frame);
  if (history_.size() > kHistory) history_.pop_front();
  const auto window = history_window();
  if (frame_index_ == 0) {
    Point result = backend.run(window);
    if (!std::isfinite(result.x) || !std::isfinite(result.y))
      throw std::runtime_error("invalid STNet observation");
    ++stnet_calls_;
    kalman_.reset(result); template_frame_ = frame; template_center_ = result;
    last_output_ = result; ++frame_index_;
    return {result, Route::Reset};
  }
  const Point previous = kalman_.position();
  const Point prediction = kalman_.predict();
  const MatchResult matched = override_match ? *override_match
      : matcher_.match(template_frame_, frame, template_center_, prediction);
  Route path = route(distance(prediction, matched.position), matched.confidence,
                     matched.border, confidence_threshold_);
  // Do not silently turn a medium-error frame into a fast-path hit if the
  // single NPU context is busy; synchronously re-anchor instead.
  if (path == Route::Deferred && pending_) path = Route::Reset;
  Point output;
  if (path == Route::Reset) {
    output = backend.run(window);
    if (!std::isfinite(output.x) || !std::isfinite(output.y))
      throw std::runtime_error("invalid STNet observation");
    ++stnet_calls_;
    kalman_.reset(output, previous); anchor_ = frame_index_;
    template_frame_ = frame; template_center_ = output;
  } else {
    if (!std::isfinite(matched.position.x) || !std::isfinite(matched.position.y))
      throw std::runtime_error("invalid matcher observation");
    const Point corrected = kalman_.update(matched.position);
    output = path == Route::Fast ? corrected : prediction;
    if (path == Route::Deferred) {
      pending_.emplace(Pending{frame_index_, prediction, previous, backend.start(window)});
      ++stnet_calls_;
    }
  }
  last_output_ = output; ++frame_index_;
  return {output, path};
}
} // namespace ktcal
