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

// #define TEST_BRUTE_FORCE

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
  nanopm::Image3b vis_nnf, vis_distance, recon;
  option.debug_dir = "./";
  nanopm::Compute(A, B, nnf, distance, option);
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

#ifdef TEST_BRUTE_FORCE
  nanopm::impl::Timer<> timer;
  timer.Start();
  nanopm::BruteForce(A, B, nnf, distance, option);
  timer.End();
  printf("nanopm::BruteForce %fms\n", timer.elapsed_msec());
  nanopm::ColorizeNnf(nnf, vis_nnf);
  nanopm::imwrite(data_dir + "nnf_bruteforce.jpg", vis_nnf);
  nanopm::ColorizeDistance(distance, vis_distance, max_d, min_d, mean, stddev);
  nanopm::imwrite(data_dir + "distance_bruteforce.jpg", vis_distance);
  nanopm::Reconstruction(nnf, option.patch_size, B, recon);
  nanopm::imwrite(data_dir + "reconstruction_bruteforce.jpg", recon);
#endif

  return 0;
}
