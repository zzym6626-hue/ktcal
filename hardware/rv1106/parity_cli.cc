// Test helper: run the actual C++ matcher on controlled binary frames.
#include "fast_path.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    std::ifstream frames(argv[2], std::ios::binary);
    ktcal::Frame before{}, after{};
    if (!frames.read(reinterpret_cast<char*>(before.data()), sizeof(before)) ||
        !frames.read(reinterpret_cast<char*>(after.data()), sizeof(after)))
      throw std::runtime_error("incomplete frames");
    ktcal::Matcher matcher(ktcal::load_matcher_weights(argv[1]));
    auto result = matcher.match(before, after, {40.1f, 30.2f}, {40.f, 30.f});
    std::cout << result.position.x << ' ' << result.position.y << ' '
              << result.confidence << ' ' << result.border << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
