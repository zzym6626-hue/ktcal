#include "rknn_sequence_backend.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: ktcal_rv1106 MATCHER.bin STNET100.rknn records-f32-chw.bin\n";
    return 2;
  }
  try {
    const ktcal::SequenceTracker tracker(ktcal::load_matcher_weights(argv[1]));
    ktcal::RknnSequenceBackend backend(argv[2]);
    std::ifstream input(argv[3], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open records");
    // Input must be little-endian float32; no implicit swapping on big-endian targets.
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1) throw std::runtime_error("little-endian target required");
    std::cout << std::setprecision(10) << "record,frame,x_px,y_px,raw_x_px,raw_y_px,tier\n";
    ktcal::Record frames;
    std::size_t records = 0, fast = 0, medium = 0, resets = 0;
    double total_ms = 0;
    while (ktcal::read_record(input, frames)) {
      const auto start = std::chrono::steady_clock::now();
      const auto global = ktcal::predict_record(frames, backend);
      const auto result = tracker.track(frames, global);
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      if (result.stats.small_fusion_calls != 0) throw std::runtime_error("low-tier fusion forbidden");
      total_ms += elapsed; fast += result.stats.tier1; medium += result.stats.tier2; resets += result.stats.tier3;
      for (std::size_t t = 0; t < ktcal::kSequence; ++t)
        std::cout << records << ',' << t << ',' << result.filtered[t][0] << ',' << result.filtered[t][1]
                  << ',' << result.raw[t][0] << ',' << result.raw[t][1] << ',' << result.tiers[t] << '\n';
      std::cerr << "record=" << records++ << " processing_ms=" << elapsed << " ms_per_frame="
                << elapsed / ktcal::kSequence << " network_calls=10\n";
    }
    if (records == 0) throw std::runtime_error("no complete 100-frame record");
    std::cerr << "records=" << records << " frames=" << records*ktcal::kSequence
              << " mean_ms_per_frame=" << total_ms/(records*ktcal::kSequence)
              << " tier1=" << fast << " tier2=" << medium << " tier3=" << resets
              << " small_fusion_calls=0; includes all views/decode/matching/KF/MA7,"
              << " excludes input collection/disk/model loading/CSV output; first run not excluded\n";
  } catch (const std::exception& error) {
    std::cerr << "ktcal_rv1106: " << error.what() << '\n'; return 1;
  }
}
