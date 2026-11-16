// Host validation adapter: exercises the actual sequence path without RKNN.
#include "sequence_path.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void read_exact(const char* path, void* data, std::size_t bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("incomplete parity input");
}
void print(const ktcal::Coordinates& values) {
  for (auto value : values) std::cout << value[0] << ' ' << value[1] << ' ';
}
}
int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(12);
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "postprocess" && argc == 4) {
      ktcal::Frame frame{};
      read_exact(argv[2], frame.data(), sizeof(frame));
      std::vector<float> logits(10 * 100 * 140);
      read_exact(argv[3], logits.data(), logits.size() * sizeof(float));
      struct Replay final : ktcal::SequenceBackend {
        std::vector<float>& logits;
        std::size_t index = 0;
        explicit Replay(std::vector<float>& values) : logits(values) {}
        ktcal::SequenceLogits run(const ktcal::Record& input) override {
          for (float value : input[0]) std::cout << value << ' ';
          ktcal::SequenceLogits value;
          for (std::size_t t = 0; t < 100; ++t) {
            auto first = logits.begin() + (index * 100 + t) * 140;
            value.x.insert(value.x.end(), first, first + 80);
            value.y.insert(value.y.end(), first + 80, first + 140);
          }
          ++index;
          return value;
        }
      } backend(logits);
      const auto coordinates = ktcal::predict_record(ktcal::Record(100, frame), backend);
      print(coordinates); print(ktcal::smooth_ma7(coordinates));
    } else if (mode == "match" && argc == 4) {
      std::array<ktcal::Frame, 2> frames{};
      read_exact(argv[3], frames.data(), sizeof(frames));
      ktcal::Matcher matcher(ktcal::load_matcher_weights(argv[2]), {37,32,true});
      const auto hit = matcher.match(frames[0], frames[1], {40.1f,30.2f}, {40,30});
      std::cout << hit.position.x << ' ' << hit.position.y << ' ' << hit.confidence
                << ' ' << hit.psr << ' ' << hit.border;
    } else if (mode == "gates" && argc == 3) {
      ktcal::SequenceGate gate;
      ktcal::SequenceKalman reference_inputs;
      gate.seed({40,30}); reference_inputs.reset({40,30});
      std::cout << "40 30 3 ";
      const float fixtures[7][5] = {{1,.8f,3,0,0}, {3,.8f,3,0,.5f}, {4,.8f,3,0,2},
                                    {5,.8f,3,0,0}, {0,.27f,3,0,0}, {0,.8f,2.54f,0,0},
                                    {0,.8f,3,1,0}};
      for (const auto& fixture : fixtures) {
        const auto previous = reference_inputs.position();
        const auto prediction = reference_inputs.predict();
        gate.predict();
        ktcal::MatchResult match{{static_cast<float>(prediction[0]) + fixture[0],
                                  static_cast<float>(prediction[1])}, fixture[1], fixture[3] != 0, fixture[2]};
        const double difference = std::hypot(prediction[0] - match.position.x, prediction[1] - match.position.y);
        const int tier = match.confidence < .28f || match.psr < 2.55f || match.border || difference >= 5 ? 3 : difference >= 3 ? 2 : 1;
        if (tier < 3) reference_inputs.update({match.position.x, match.position.y});
        const auto posterior = reference_inputs.position();
        ktcal::Pixel global{static_cast<float>(posterior[0] + fixture[4]), static_cast<float>(posterior[1])};
        if (tier == 1 && std::stoi(argv[2])) { global[0] += 1000; global[1] += 1000; }
        if (tier == 3 || (tier == 2 && std::hypot(posterior[0]-global[0],posterior[1]-global[1]) > 1))
          reference_inputs.reset(global, previous);
        else if (tier == 2) reference_inputs.update(global);
        const auto result = gate.correct(match, global);
        std::cout << result.position[0] << ' ' << result.position[1] << ' ' << result.tier << ' ';
      }
      if (gate.stats().small_fusion_calls != 0) throw std::runtime_error("low fusion forbidden");
    } else if (mode == "track" && argc == 5) {
      ktcal::Record record(100);
      read_exact(argv[3], record.data(), record.size() * sizeof(ktcal::Frame));
      std::array<float, 200> values{};
      read_exact(argv[4], values.data(), sizeof(values));
      ktcal::Coordinates globals{};
      for (std::size_t t = 0; t < 100; ++t) globals[t] = {values[2*t], values[2*t+1]};
      const ktcal::SequenceTracker tracker(ktcal::load_matcher_weights(argv[2]));
      for (int repeat = 0; repeat < 2; ++repeat) {
        const auto result = tracker.track(record, globals);
        for (std::size_t t = 0; t < 100; ++t)
          std::cout << result.raw[t][0] << ' ' << result.raw[t][1] << ' ' << result.tiers[t] << ' '
                    << result.filtered[t][0] << ' ' << result.filtered[t][1] << ' ';
      }
    } else {
      throw std::invalid_argument("usage: sequence_parity postprocess|match|gates|track INPUTS...");
    }
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
