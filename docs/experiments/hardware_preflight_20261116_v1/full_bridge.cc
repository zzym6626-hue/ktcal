// Validation-only replay of ONNX logits through production C++ sequence code.
#include "sequence_path.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

void read(const char* path, void* data, std::size_t size) {
  std::ifstream input(path, std::ios::binary);
  if (!input.read(static_cast<char*>(data),size) || input.peek()!=std::char_traits<char>::eof())
    throw std::runtime_error("invalid exact-size replay input");
}
struct Replay final : ktcal::SequenceBackend {
  ktcal::Record views;
  std::vector<float> logits;
  std::size_t index=0;
  Replay(const char* view_path,const char* logit_path):views(1000),logits(10*100*140) {
    read(view_path,views.data(),views.size()*sizeof(ktcal::Frame));
    read(logit_path,logits.data(),logits.size()*sizeof(float));
  }
  ktcal::SequenceLogits run(const ktcal::Record& input) override {
    if(index>=10 || std::memcmp(input.data(),views.data()+index*100,100*sizeof(ktcal::Frame)))
      throw std::runtime_error("C++ and Python transformed raw inputs differ");
    ktcal::SequenceLogits output;
    for(std::size_t t=0;t<100;++t) {
      auto first=logits.begin()+(index*100+t)*140;
      output.x.insert(output.x.end(),first,first+80);
      output.y.insert(output.y.end(),first+80,first+140);
    }
    ++index;return output;
  }
};
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::invalid_argument("matcher, raw, views, logits, output required");
    ktcal::Record raw(100);read(argv[2],raw.data(),raw.size()*sizeof(ktcal::Frame));
    Replay backend(argv[3],argv[4]);
    ktcal::SequenceTracker tracker(ktcal::load_matcher_weights(argv[1]));
    auto start=std::chrono::steady_clock::now();
    const auto global=ktcal::predict_record(raw,backend);
    const auto stnet_filtered=ktcal::smooth_ma7(global);
    if(backend.index!=10)throw std::runtime_error("all ten views required");
    const auto result=tracker.track(raw,global);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    auto perturbed=global;
    for(std::size_t t=1;t<100;++t)if(result.tiers[t]==1) {
      perturbed[t][0]+=10000;perturbed[t][1]-=10000;
    }
    const auto invariant=tracker.track(raw,perturbed);
    if(result.raw!=invariant.raw || result.tiers!=invariant.tiers || result.stats.small_fusion_calls!=0)
      throw std::runtime_error("low-tier STNet invariance failed");
    std::ofstream output(argv[5],std::ios::binary);
    for(std::size_t t=0;t<100;++t) {
      const double row[]={global[t][0],global[t][1],result.raw[t][0],result.raw[t][1],
                          result.filtered[t][0],result.filtered[t][1],double(result.tiers[t]),
                          stnet_filtered[t][0],stnet_filtered[t][1]};
      output.write(reinterpret_cast<const char*>(row),sizeof(row));
    }
    if(!output)throw std::runtime_error("output write failed");
    const auto& s=result.stats;
    std::cout<<"{\"tier1\":"<<s.tier1<<",\"tier2\":"<<s.tier2<<",\"tier3\":"<<s.tier3
             <<",\"medium_refinements\":"<<s.medium_refinements<<",\"medium_rollbacks\":"<<s.medium_rollbacks
             <<",\"fusion_calls\":"<<s.fusion_calls<<",\"small_fusion_calls\":"<<s.small_fusion_calls
             <<",\"stnet_requests\":"<<s.stnet_requests<<",\"border_peaks\":"<<s.border_peaks
             <<",\"replay_and_cpp_ms\":"<<ms<<",\"low_global_invariant\":true,\"all_raw_views_equal\":true}\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
