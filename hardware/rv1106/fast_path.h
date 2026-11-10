#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <optional>
#include <vector>

namespace ktcal {

constexpr int kWidth = 80;
constexpr int kHeight = 60;
constexpr int kHistory = 16;
using Frame = std::array<float, 2 * kWidth * kHeight>; // planar CHW, raw event counts
using Window = std::array<Frame, kHistory>;            // oldest first

struct Point { float x, y; }; // 80x60 pixel coordinates, NOT normalized
enum class Route { Fast, Deferred, Reset };
Route route(float error_px, float confidence, bool border);

struct MatcherWeights {
  std::uint32_t channels = 0;
  std::array<float, 18> depthwise{}; // PyTorch [2,1,3,3]
  std::vector<float> pointwise;      // PyTorch [channels,2,1,1]
};
MatcherWeights load_matcher_weights(const char* file);

struct MatchResult { Point position; float confidence; bool border; };
class Matcher {
 public:
  explicit Matcher(MatcherWeights weights);
  MatchResult match(const Frame& previous, const Frame& current,
                    Point template_center, Point search_center) const;
 private:
  MatcherWeights weights_;
};

class Kalman {
 public:
  void reset(Point observation, std::optional<Point> previous = std::nullopt);
  Point predict();
  Point update(Point observation);
  Point position() const;
 private:
  std::array<double, 4> state_{};
  std::array<std::array<double, 4>, 4> covariance_{};
};

// Backend must outlive Tracker (a deferred future may still refer to it).
// run is synchronous; start must return a future for the SAME 16-frame input.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual Point run(const Window& frames) = 0;
  virtual std::future<Point> start(Window frames) = 0;
};

struct Output { Point position; Route path; };
class Tracker {
 public:
  explicit Tracker(Matcher matcher);
  Output tick(const Frame& frame, Backend& backend);
  // An externally supplied observation is useful when matching runs on a separate CPU core.
  Output tick_with_match(const Frame& frame, Backend& backend, MatchResult match);
  // Wait for the tail correction and settle stats/state; does not rewrite outputs.
  void finish();
  std::size_t stnet_calls() const { return stnet_calls_; }
  std::size_t async_applied() const { return async_applied_; }
  std::size_t async_discarded() const { return async_discarded_; }
 private:
  Output process(const Frame& frame, Backend& backend,
                 std::optional<MatchResult> override_match);
  Window history_window() const;
  void collect();
  Matcher matcher_;
  Kalman kalman_;
  std::deque<Frame> history_;
  Frame template_frame_{};
  Point template_center_{};
  Point last_output_{};
  struct Pending {
    std::size_t frame;
    Point predicted;
    Point previous;
    std::future<Point> future;
  };
  std::optional<Pending> pending_;
  std::size_t frame_index_ = 0, anchor_ = 0, stnet_calls_ = 0;
  std::size_t async_applied_ = 0, async_discarded_ = 0;
};
} // namespace ktcal
