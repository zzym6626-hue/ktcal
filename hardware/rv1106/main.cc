#include "fast_path.h"
#include "rknn_backend.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: ktcal_rv1106 MATCHER.bin STNET.rknn frames-f32-chw.bin INPUT_DIVISOR\n";
    return 2;
  }
  try {
    const float divisor = std::stof(argv[4]);
    ktcal::RknnBackend backend(argv[2], divisor); // outlives Tracker / async task
    ktcal::Tracker tracker(ktcal::Matcher(ktcal::load_matcher_weights(argv[1])));
    std::ifstream frames(argv[3], std::ios::binary);
    if (!frames) throw std::runtime_error("cannot open frame input");
    ktcal::Frame frame{};
    std::cout << "frame,x_px,y_px,path,tick_ms\n";
    std::size_t count = 0;
    while (frames.read(reinterpret_cast<char*>(frame.data()), sizeof(frame))) {
      auto start = std::chrono::steady_clock::now();
      auto output = tracker.tick(frame, backend);
      auto stop = std::chrono::steady_clock::now();
      const char* path = output.path == ktcal::Route::Fast ? "fast" :
                         output.path == ktcal::Route::Deferred ? "deferred" : "reset";
      std::cout << count++ << ',' << output.position.x << ',' << output.position.y
                << ',' << path << ','
                << std::chrono::duration<double, std::milli>(stop-start).count() << '\n';
    }
    if (frames.gcount() != 0 || count == 0)
      throw std::runtime_error("input must contain complete 2x60x80 float32 frames");
    tracker.finish(); // settle the final async task before reporting counters
    std::cerr << "frames=" << count << " stnet_calls=" << tracker.stnet_calls()
              << " async_applied=" << tracker.async_applied()
              << " async_discarded=" << tracker.async_discarded() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "ktcal_rv1106: " << error.what() << '\n';
    return 1;
  }
}
