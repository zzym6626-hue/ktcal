#include "fast_path.h"

#include <cassert>
#include <cmath>
#include <future>
#include <limits>
#include <cstdio>
#include <stdexcept>

using namespace ktcal;

static bool near(float a, float b, float tolerance = 1e-3f) {
  return std::abs(a - b) < tolerance;
}

struct FakeBackend final : Backend {
  int synchronous = 0;
  int asynchronous = 0;
  std::size_t window_length = 0;
  bool delay = false;
  std::optional<std::promise<Point>> delayed_result;
  Point synchronous_result{40, 30};
  Point asynchronous_result{40.5f, 30};
  Point run(const Window& frames) override {
    ++synchronous;
    window_length = frames.size();
    return synchronous_result;
  }
  std::future<Point> start(Window frames) override {
    ++asynchronous;
    window_length = frames.size();
    if (delay) {
      delayed_result.emplace();
      return delayed_result->get_future();
    }
    std::promise<Point> promise;
    promise.set_value(asynchronous_result);
    return promise.get_future();
  }
};

int main() {
  assert(route(2.99f, 0.8f, false) == Route::Fast);
  assert(route(3.0f, 0.8f, false) == Route::Deferred);
  assert(route(5.0f, 0.8f, false) == Route::Reset);
  assert(route(0, 0.49f, false) == Route::Reset);
  assert(route(0, 0.9f, true) == Route::Reset);
  assert(route(0, std::numeric_limits<float>::quiet_NaN(), false) == Route::Reset);

  Kalman filter;
  filter.reset({10, 20});
  assert(near(filter.predict().x, 10));
  const Point corrected = filter.update({12, 20});
  assert(corrected.x > 10 && corrected.x < 12);
  filter.reset({8, 11}, Point{6, 11});
  assert(near(filter.predict().x, 10));
  assert(near(filter.predict().y, 11));

  MatcherWeights weights;
  weights.channels = 2;
  weights.depthwise.fill(0);
  weights.depthwise[4] = 1;
  weights.depthwise[13] = 1;
  weights.pointwise = {1, 0, 0, 1};
  Matcher matcher(weights);
  Frame prior{}, shifted{}, blank{};
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth - 1; ++x) {
      const float v = static_cast<float>(((x * 61 + y * 73 + x * y * 13) % 251) + 1);
      prior[y * kWidth + x] = v;
      shifted[y * kWidth + x + 1] = v;
    }
  }
  const MatchResult hit = matcher.match(prior, shifted, {40, 30}, {40, 30});
  assert(hit.confidence > 0.5f);
  assert(std::abs(hit.position.x - 41) <= 0.51f);
  assert(std::abs(hit.position.y - 30) <= 0.51f);
  assert(!hit.border);
  assert(matcher.match(prior, blank, {40, 30}, {40, 30}).confidence == 0);

  FakeBackend backend;
  Tracker tracker(matcher);
  const Output initial = tracker.tick(prior, backend);
  assert(initial.path == Route::Reset && backend.synchronous == 1);
  assert(backend.window_length == 16);
  const Output normal = tracker.tick(shifted, backend);
  assert(normal.path == Route::Fast);
  assert(backend.synchronous == 1 && backend.asynchronous == 0);
  const Output missing = tracker.tick(blank, backend);
  assert(missing.path == Route::Reset && backend.synchronous == 2);

  // Controlled observations isolate the three-tier state machine from NCC.
  Tracker gate_tracker(matcher);
  gate_tracker.tick(prior, backend);
  const Output medium = gate_tracker.tick_with_match(shifted, backend,
                                                   {{43, 30}, 0.8f, false});
  assert(medium.path == Route::Deferred && backend.asynchronous == 1);
  assert(near(medium.position.x, 40)); // emitted prediction, not future observation
  const Output next = gate_tracker.tick_with_match(shifted, backend,
                                                 {{40, 30}, 0.8f, false});
  assert(next.path == Route::Fast);
  assert(gate_tracker.async_applied() == 1);
  assert(gate_tracker.stnet_calls() == 2); // seed plus medium

  FakeBackend busy;
  busy.delay = true;
  Tracker busy_tracker(matcher);
  busy_tracker.tick(prior, busy);
  const Output first_medium = busy_tracker.tick_with_match(shifted, busy,
                                                            {{43, 30}, 0.8f, false});
  assert(first_medium.path == Route::Deferred);
  const Output second_medium = busy_tracker.tick_with_match(shifted, busy,
                                                             {{48, 30}, 0.8f, false});
  // With an NPU job still busy, never misreport the next medium frame as a fast hit.
  assert(second_medium.path == Route::Reset && busy.synchronous == 2);
  busy.delayed_result->set_value({40, 30});
  busy_tracker.tick_with_match(shifted, busy, {{40, 30}, 0.1f, false});
  assert(busy_tracker.async_discarded() == 1);

  // An async correction must derive its reset velocity from the PREVIOUS
  // frame's posterior, not the corrected frame's prior prediction.
  FakeBackend moving;
  Tracker moving_tracker(matcher);
  moving_tracker.tick(prior, moving);
  moving.synchronous_result = {42, 30};
  moving_tracker.tick_with_match(shifted, moving, {{40, 30}, 0.1f, false});
  moving.asynchronous_result = {48, 30};
  const Output moving_medium = moving_tracker.tick_with_match(
      shifted, moving, {{47, 30}, 0.8f, false});
  assert(moving_medium.path == Route::Deferred);
  // previous=42, corrected=48 => next prediction=54 (not 52).
  const Output next_moving = moving_tracker.tick_with_match(
      shifted, moving, {{58, 30}, 0.8f, false});
  assert(next_moving.path == Route::Deferred);
  assert(near(next_moving.position.x, 54));
  moving_tracker.finish();

  // A superseded job's NPU error is irrelevant to an already successful reset.
  FakeBackend stale;
  stale.delay = true;
  Tracker stale_tracker(matcher);
  stale_tracker.tick(prior, stale);
  stale_tracker.tick_with_match(shifted, stale, {{43, 30}, 0.8f, false});
  stale_tracker.tick_with_match(shifted, stale, {{40, 30}, 0.1f, false});
  stale.delayed_result->set_exception(
      std::make_exception_ptr(std::runtime_error("superseded NPU failure")));
  const Output recovered = stale_tracker.tick_with_match(
      shifted, stale, {{40, 30}, 0.8f, false});
  assert(recovered.path == Route::Fast && stale_tracker.async_discarded() == 1);

  // EOF must account for the final outstanding job without changing emitted frames.
  FakeBackend final;
  final.delay = true;
  Tracker final_tracker(matcher);
  final_tracker.tick(prior, final);
  final_tracker.tick_with_match(shifted, final, {{43, 30}, 0.8f, false});
  final.delayed_result->set_value({40.5f, 30});
  final_tracker.finish();
  assert(final_tracker.async_applied() == 1);
  assert(final_tracker.async_applied() + final_tracker.async_discarded() == 1);
}
