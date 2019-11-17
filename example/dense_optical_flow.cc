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
  nanopm::Compute(A, B, nnf, distance, option);

  nanopm::Image3b vis_nnf, vis_distance;
  nanopm::ColorizeNnf(nnf, vis_nnf);
  nanopm::imwrite(data_dir + "nnf.jpg", vis_nnf);
  nanopm::ColorizeDistance(distance, vis_distance);
  nanopm::imwrite(data_dir + "distance.jpg", vis_distance);

  return 0;
}
