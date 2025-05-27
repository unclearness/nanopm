/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#ifdef NANOPM_USE_STB
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#include "nanopm.h"

#include <numeric>
#include <vector>

namespace {
constexpr bool output_debug_images =
    false;  // Set to true to output debug images
constexpr bool test_brute_force =
    true;  // Set to true to test brute force method
}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  nanopm::impl::Timer<> timer;

  std::string data_dir = "../data/scenes2005/Art/";

  nanopm::Image3b A = nanopm::imread<nanopm::Image3b>(
      data_dir + "view1.png", nanopm::ImreadModes::IMREAD_COLOR);
  nanopm::Image3b B = nanopm::imread<nanopm::Image3b>(
      data_dir + "view5.png", nanopm::ImreadModes::IMREAD_COLOR);

  nanopm::Image2f nnf;
  nanopm::Image1f distance;
  nanopm::Option option;
  nanopm::Image3b vis_nnf, vis_distance, recon;

  if (output_debug_images) {
    option.verbose = true;
    option.debug_dir = "./";
  } else {
    option.verbose = false;
    option.debug_dir = "";
  }

  timer.Start();
  nanopm::Compute(A, B, nnf, distance, option);
  timer.End();
  printf("nanopm::Compute %fms\n", timer.elapsed_msec());
  nanopm::ColorizeNnf(nnf, vis_nnf);
  nanopm::imwrite(data_dir + "nnf.jpg", vis_nnf);
  float mean, stddev;
  float max_d = 17000.0f;
  float min_d = 50.0f;
  nanopm::ColorizeDistance(distance, vis_distance, max_d, min_d, mean, stddev);
  printf("distance mean %f, stddev %f\n", mean, stddev);
  nanopm::imwrite(data_dir + "distance.jpg", vis_distance);
  nanopm::Reconstruction(nnf, option.patch_size, B, recon);
  nanopm::imwrite(data_dir + "reconstruction.jpg", recon);

  if (test_brute_force) {
    timer.Start();
    nanopm::BruteForce(A, B, nnf, distance, option);
    timer.End();
    printf("nanopm::BruteForce %fms\n", timer.elapsed_msec());
    nanopm::ColorizeNnf(nnf, vis_nnf);
    nanopm::imwrite(data_dir + "nnf_bruteforce.jpg", vis_nnf);
    nanopm::ColorizeDistance(distance, vis_distance, max_d, min_d, mean,
                             stddev);
    nanopm::imwrite(data_dir + "distance_bruteforce.jpg", vis_distance);
    nanopm::Reconstruction(nnf, option.patch_size, B, recon);
    nanopm::imwrite(data_dir + "reconstruction_bruteforce.jpg", recon);
  }
  return 0;
}
