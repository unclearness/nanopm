/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#ifdef NANOPM_USE_STB
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "nanopm.h"

#include <chrono>  //NOLINT
#include <numeric>
#include <vector>

// #define TEST_BRUTE_FORCE

template <typename T = double>
class Timer {
  std::chrono::system_clock::time_point start_t_, end_t_;
  T elapsed_msec_{-1};
  size_t history_num_{30};
  std::vector<T> history_;

 public:
  Timer() {}
  ~Timer() {}
  explicit Timer(size_t history_num) : history_num_(history_num) {}

  std::chrono::system_clock::time_point start_t() const { return start_t_; }
  std::chrono::system_clock::time_point end_t() const { return end_t_; }

  void Start() { start_t_ = std::chrono::system_clock::now(); }
  void End() {
    end_t_ = std::chrono::system_clock::now();
    elapsed_msec_ = static_cast<T>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_t_ - start_t_)
            .count() *
        0.001);

    history_.push_back(elapsed_msec_);
    if (history_num_ < history_.size()) {
      history_.erase(history_.begin());
    }
  }
  T elapsed_msec() const { return elapsed_msec_; }
  T average_msec() const {
    return static_cast<T>(std::accumulate(history_.begin(), history_.end(), 0) /
                          history_.size());
  }
};

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/scenes2005/Art/";

  nanopm::Image3b A = nanopm::imread<nanopm::Image3b>(
      data_dir + "view1.png", nanopm::ImreadModes::IMREAD_COLOR);
  nanopm::Image3b B = nanopm::imread<nanopm::Image3b>(
      data_dir + "view5.png", nanopm::ImreadModes::IMREAD_COLOR);

  nanopm::Image2f nnf;
  nanopm::Image1f distance;
  nanopm::Option option;
  nanopm::Image3b vis_nnf, vis_distance;
  option.debug_dir = "./";
  Timer<> timer;
  timer.Start();
  nanopm::Compute(A, B, nnf, distance, option);
  timer.End();
  printf("nanopm::Compute %fms\n", timer.elapsed_msec());
  nanopm::ColorizeNnf(nnf, vis_nnf);
  nanopm::imwrite(data_dir + "nnf.jpg", vis_nnf);
  nanopm::ColorizeDistance(distance, vis_distance);
  nanopm::imwrite(data_dir + "distance.jpg", vis_distance);

#ifdef TEST_BRUTE_FORCE
  timer.Start();
  nanopm::BruteForce(A, B, nnf, distance, option);
  timer.End();
  printf("nanopm::BruteForce %fms\n", timer.elapsed_msec());
  nanopm::ColorizeNnf(nnf, vis_nnf);
  nanopm::imwrite(data_dir + "nnf_bruteforce.jpg", vis_nnf);
  nanopm::ColorizeDistance(distance, vis_distance);
  nanopm::imwrite(data_dir + "distance_bruteforce.jpg", vis_distance);
#endif

  return 0;
}
