/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "nanopm.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/";

  nanopm::Image3b A = nanopm::imread<nanopm::Image3b>(
      data_dir + "1.jpg", nanopm::ImreadModes::IMREAD_COLOR);
  nanopm::Image3b B = nanopm::imread<nanopm::Image3b>(
      data_dir + "2.jpg", nanopm::ImreadModes::IMREAD_COLOR);

  nanopm::Image2f nnf;
  nanopm::Image1f distance;
  nanopm::Option option;
  option.max_iter = 5;
  option.patch_size = 15;
  option.w = std::pow(2.0f, option.max_iter-1);
  nanopm::Compute(A, B, nnf, distance, option);

  nanopm::Image3b vis_nnf, vis_distance;
  nanopm::ColorizeNnf(nnf, vis_nnf);
  nanopm::imwrite(data_dir + "nnf.jpg", vis_nnf);
  nanopm::ColorizeDistance(distance, vis_distance);
  nanopm::imwrite(data_dir + "distance.jpg", vis_distance);

  return 0;
}
