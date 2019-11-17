/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <array>
#include <random>
#include <unordered_map>
#include <vector>

#ifdef NANOPM_USE_STB
#include "stb_image.h"
#include "stb_image_write.h"
#endif

#ifdef NANOPM_USE_LODEPNG
#include "lodepng/lodepng.h"
#endif

#ifdef NANOPM_USE_TINYCOLORMAP
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4067)
#endif
#include "tinycolormap/include/tinycolormap.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

#ifdef NANOPM_USE_OPENCV
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#endif

namespace nanopm {

/* start of Image class definition */

#ifdef NANOPM_USE_OPENCV
template <typename T>
using Image = cv::Mat_<T>;

using Image1b = cv::Mat1b;
using Image3b = cv::Mat3b;
using Image1w = cv::Mat1w;
using Image1i = cv::Mat1i;
using Image1f = cv::Mat1f;
using Image2f = cv::Mat2f;
using Image3f = cv::Mat3f;

using Vec1v = unsigned char;
using Vec1f = float;
using Vec1i = int;
using Vec1w = std::uint16_t;
using Vec2i = cv::Vec2i;
using Vec2f = cv::Vec2f;
using Vec3f = cv::Vec3f;
using Vec3b = cv::Vec3b;

using ImreadModes = cv::ImreadModes;

template <typename T>
inline bool imwrite(const std::string& filename, const T& img,
                    const std::vector<int>& params = std::vector<int>()) {
  return cv::imwrite(filename, img, params);
}

template <typename T>
inline T imread(const std::string& filename,
                int flags = ImreadModes::IMREAD_COLOR) {
  return cv::imread(filename, flags);
}

template <typename T, typename TT>
inline void Init(Image<T>* image, int width, int height, TT val) {
  if (image->cols == width && image->rows == height) {
    image->setTo(val);
  } else {
    if (val == TT(0)) {
      *image = Image<T>::zeros(height, width);
    } else {
      *image = Image<T>::ones(height, width) * val;
    }
  }
}

template <typename T, typename TT>
bool ConvertTo(const Image<T>& src, Image<TT>* dst, float scale = 1.0f) {
  src.convertTo(*dst, dst->type(), scale);

  return true;
}

#else

template <typename TT, int N>
using Vec = std::array<TT, N>;

using Vec1f = Vec<float, 1>;
using Vec1i = std::array<int, 1>;
using Vec1w = std::array<std::uint16_t, 1>;
using Vec1b = std::array<unsigned char, 1>;
using Vec3b = std::array<unsigned char, 3>;
using Vec3f = std::array<float, 3>;

template <typename T>
class Image {
 private:
  int bit_depth_{sizeof(typename T::value_type)};
  int channels_{std::tuple_size<T>::value};
  int width_{-1};
  int height_{-1};
  std::shared_ptr<std::vector<T> > data_{nullptr};

  void Init(int width, int height) {
    if (width < 1 || height < 1) {
      LOGE("wrong width or height\n");
      return;
    }

    width_ = width;
    height_ = height;
    data_->resize(height_ * width_);
    data = reinterpret_cast<unsigned char*>(data_->data());
    rows = height;
    cols = width;

    channels_ = static_cast<int>((*data_)[0].size());
  }

  void Init(int width, int height, typename T::value_type val) {
    if (width < 1 || height < 1) {
      LOGE("wrong width or height\n");
      return;
    }

    Init(width, height);

    this->setTo(val);
  }

 public:
  Image() : data_(new std::vector<T>) {}
  ~Image() {}
  int channels() const { return channels_; }

  int rows;
  int cols;
  unsigned char* data;

  bool empty() const {
    if (width_ < 0 || height_ < 0 || data_->empty()) {
      return true;
    }
    return false;
  }

  template <typename TT>
  TT& at(int y, int x) {
    return *(reinterpret_cast<TT*>(data_->data()) + (y * cols + x));
  }
  template <typename TT>
  const TT& at(int y, int x) const {
    return *(reinterpret_cast<TT*>(data_->data()) + (y * cols + x));
  }

  void setTo(typename T::value_type val) {
    for (auto& v : *data_) {
      for (auto& vv : v) {
        vv = val;
      }
    }
  }

  static Image<T> zeros(int height, int width) {
    Image<T> tmp;
    tmp.Init(width, height, static_cast<typename T::value_type>(0));
    return tmp;
  }

#ifdef NANOPM_USE_STB
  bool Load(const std::string& path) {
    unsigned char* in_pixels_tmp;
    int width;
    int height;
    int bpp;

    if (bit_depth_ == 2) {
      in_pixels_tmp = reinterpret_cast<unsigned char*>(
          stbi_load_16(path.c_str(), &width, &height, &bpp, channels_));
    } else if (bit_depth_ == 1) {
      in_pixels_tmp = stbi_load(path.c_str(), &width, &height, &bpp, channels_);
    } else {
      LOGE("Load() for bit_depth %d and channel %d is not supported\n",
           bit_depth_, channels_);
      return false;
    }

    if (bpp != channels_) {
      delete in_pixels_tmp;
      LOGE("desired channel %d, actual %d\n", channels_, bpp);
      return false;
    }

    Init(width, height);

    std::memcpy(data_->data(), in_pixels_tmp, sizeof(T) * width_ * height_);
    delete in_pixels_tmp;

    return true;
  }

#ifdef NANOPM_USE_LODEPNG
  // https://github.com/lvandeve/lodepng/issues/74#issuecomment-405049566
  bool WritePng16Bit1Channel(const std::string& path) const {
    if (bit_depth_ != 2 || channels_ != 1) {
      LOGE("WritePng16Bit1Channel invalid bit_depth %d or channel %d\n",
           bit_depth_, channels_);
      return false;
    }
    std::vector<unsigned char> data_8bit;
    data_8bit.resize(width_ * height_ * 2);  // 2 bytes per pixel
    const int kMostMask = 0b1111111100000000;
    const int kLeastMask = ~kMostMask;
    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        std::uint16_t d = this->at<std::uint16_t>(y, x);  // At(*this, x, y, 0);
        data_8bit[2 * width_ * y + 2 * x + 0] = static_cast<unsigned char>(
            (d & kMostMask) >> 8);  // most significant
        data_8bit[2 * width_ * y + 2 * x + 1] =
            static_cast<unsigned char>(d & kLeastMask);  // least significant
      }
    }
    unsigned error = lodepng::encode(
        path, data_8bit, width_, height_, LCT_GREY,
        16);  // note that the LCT_GREY and 16 parameters are of the std::vector
              // we filled in, lodepng will choose its output format itself
              // based on the colors it gets, it will choose 16-bit greyscale in
              // this case though because of the pixel data we feed it
    if (error != 0) {
      LOGE("lodepng::encode errorcode: %d\n", error);
      return false;
    }
    return true;
  }
#endif

  bool WritePng(const std::string& path) const {
#ifdef NANOPM_USE_LODEPNG
    if (bit_depth_ == 2 && channels_ == 1) {
      return WritePng16Bit1Channel(path);
    }
#endif

    if (bit_depth_ != 1) {
      LOGE("1 byte per channel is required to save by stb_image: actual %d\n",
           bit_depth_);
      return false;
    }

    if (width_ < 0 || height_ < 0) {
      LOGE("image is empty\n");
      return false;
    }

    int ret = stbi_write_png(path.c_str(), width_, height_, channels_,
                             data_->data(), width_ * sizeof(T));
    return ret != 0;
  }

  bool WriteJpg(const std::string& path) const {
    if (bit_depth_ != 1) {
      LOGE("1 byte per channel is required to save by stb_image: actual %d\n",
           bit_depth_);
      return false;
    }

    if (width_ < 0 || height_ < 0) {
      LOGE("image is empty\n");
      return false;
    }

    if (channels_ > 3) {
      LOGW("alpha channel is ignored to save as .jpg. channels(): %d\n",
           channels_);
    }

    // JPEG does ignore alpha channels in input data; quality is between 1
    // and 100. Higher quality looks better but results in a bigger image.
    const int max_quality{100};

    int ret = stbi_write_jpg(path.c_str(), width_, height_, channels_,
                             data_->data(), max_quality);
    return ret != 0;
  }
#else
  bool Load(const std::string& path) {
    (void)path;
    LOGE("can't load image with this configuration\n");
    return false;
  }

  bool WritePng(const std::string& path) const {
    (void)path;
    LOGE("can't write image with this configuration\n");
    return false;
  }

  bool WriteJpg(const std::string& path) const {
    (void)path;
    LOGE("can't write image with this configuration\n");
    return false;
  }
#endif

  void copyTo(Image<T>& dst) const {  // NOLINT
    if (dst.cols != cols || dst.rows != rows) {
      dst = Image<T>::zeros(rows, cols);
    }
    std::memcpy(dst.data_->data(), data_->data(), sizeof(T) * rows * cols);
  }
};

using Image1b = Image<Vec1b>;  // For gray image.
using Image3b = Image<Vec3b>;  // For color image. RGB order.
using Image1w = Image<Vec1w>;  // For depth image with 16 bit (unsigned
                               // short) mm-scale format
using Image1i = Image<Vec1i>;  // For face visibility. face id is within int32_t
using Image1f = Image<Vec1f>;  // For depth image with any scale
using Image3f = Image<Vec3f>;  // For normal or point cloud. XYZ order.

enum ImreadModes {
  IMREAD_UNCHANGED = -1,
  IMREAD_GRAYSCALE = 0,
  IMREAD_COLOR = 1,
  IMREAD_ANYDEPTH = 2,
  IMREAD_ANYCOLOR = 4,
  IMREAD_LOAD_GDAL = 8,
  IMREAD_REDUCED_GRAYSCALE_2 = 16,
  IMREAD_REDUCED_COLOR_2 = 17,
  IMREAD_REDUCED_GRAYSCALE_4 = 32,
  IMREAD_REDUCED_COLOR_4 = 33,
  IMREAD_REDUCED_GRAYSCALE_8 = 64,
  IMREAD_REDUCED_COLOR_8 = 65,
  IMREAD_IGNORE_ORIENTATION = 128,
};

template <typename T>
inline void Init(Image<T>* image, int width, int height) {
  if (image->cols != width || image->rows != height) {
    *image = Image<T>::zeros(height, width);
  }
}

template <typename T>
inline void Init(Image<T>* image, int width, int height,
                 typename T::value_type val) {
  if (image->cols != width || image->rows != height) {
    *image = Image<T>::zeros(height, width);
  }
  image->setTo(val);
}

template <typename T>
inline bool imwrite(const std::string& filename, const T& img,
                    const std::vector<int>& params = std::vector<int>()) {
  (void)params;
  if (filename.size() < 4) {
    return false;
  }

  size_t ext_i = filename.find_last_of(".");
  std::string extname = filename.substr(ext_i, filename.size() - ext_i);
  if (extname == ".png" || extname == ".PNG") {
    return img.WritePng(filename);
  } else if (extname == ".jpg" || extname == ".jpeg" || extname == ".JPG" ||
             extname == ".JPEG") {
    return img.WriteJpg(filename);
  }

  LOGE(
      "acceptable extention is .png, .jpg or .jpeg. this extention is not "
      "supported: %s\n",
      filename.c_str());
  return false;
}

template <typename T>
inline T imread(const std::string& filename,
                int flags = ImreadModes::IMREAD_COLOR) {
  (void)flags;
  T loaded;
  loaded.Load(filename);
  return loaded;
}

template <typename T, typename TT>
bool ConvertTo(const Image<T>& src, Image<TT>* dst, float scale = 1.0f) {
  if (src.channels() != dst->channels()) {
    LOGE("ConvertTo failed src channel %d, dst channel %d\n", src.channels(),
         dst->channels());
    return false;
  }

  Init(dst, src.cols, src.rows);

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      for (int c = 0; c < dst->channels(); c++) {
        dst->template at<TT>(y, x)[c] = static_cast<typename TT::value_type>(
            scale * src.template at<T>(y, x)[c]);
      }
    }
  }

  return true;
}

#endif

/* end of Image class definition */

/* declation of interface */

enum class InitType {
  RANDOM,         // independent pixel-wise random initialization by uniform
                  // distribution.
  INITIAL,        // use initial.
  INITIAL_RANDOM  // use initial and random. described in "3.1 Initialization"
                  // of the original paper.
};

enum class DistanceType {
  SSD,  // Sum of Squared Difference(SSD)
  SAD   // Sum of Abusolute Difference(SAD)
};

struct Option {
  int patch_size = 7;
  int max_iter = 5;

  float w = 7.0f;
  float alpha = 0.5f;

  InitType init_type = InitType::RANDOM;
  Image2f initial;
  int initial_random_iter = 5;

  size_t random_seed = 0;  // for repeatability

  DistanceType distance_type = DistanceType::SSD;

  bool verbose = true;
};

bool Compute(const Image3b& A, const Image3b& B, Image2f& nnf,
             Image1f& distance, const Option& option);

float CalcDistance(const Image3b& A, const Image3b& B, const Image2f& nnf,
                   Image1f& distance, int x, int y, const Option& option);

bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                  int B_y, int half_patch_size, DistanceType distance_type,
                  float& distance);

bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                  int B_y, int half_patch_size, DistanceType distance_type,
                  float& distance, float current_min);

bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x, int B_y,
         int half_patch_size, float& val);

bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x, int B_y,
         int half_patch_size, float& val, float current_min);

#if 0
				float SSD(const Image1b& A, int A_x, int A_y, const Image1b& B, int B_x,
          int B_y, int half_patch_size);

#endif  // 0

class DistanceCache {
  // data[i][j]: distance between index i of A and index j of B
  // std::vector<std::unordered_map<int, float>> data_;

  const Image3b* A_;
  const Image3b* B_;
  int half_patch_size_;
  Image1f min_distance_;
  DistanceType distance_type_;

 public:
  DistanceCache() = delete;
  DistanceCache(const DistanceCache& src) = delete;
  ~DistanceCache() = default;
  DistanceCache(const Image3b& A, const Image3b& B,
                const DistanceType& distance_type, int half_patch_size)
      : A_(&A),
        B_(&B),
        distance_type_(distance_type),
        half_patch_size_(half_patch_size) {
    // data_.resize(A_->cols * A_->rows);

    min_distance_ = Image1f::zeros(A_->rows, A_->cols);
    min_distance_.setTo(-1.0f);
  }
#if 0
				  DistanceCache(const Image1b& A_gray, const Image1b& B_gray,
                const DistanceType& distance_type, int patch_size)
      : A_gray_(&A_gray), B_gray_(&B_gray), distance_type_(distance_type), patch_size_(patch_size) {
    data_.resize(A_gray_->cols * A_gray_->rows);
  }
#endif  // 0

  const Image1f& min_distance() { return min_distance_; }
  int half_patch_size() { return half_patch_size_; }
  bool query(int A_x, int A_y, int x_offset, int y_offset, float& dist,
             bool& updated) {
    // int A_index = A_y * A_->cols + A_x;
    // int B_index = B_y * B_->cols + B_x;
    int B_x = A_x + x_offset;
    int B_y = A_y + y_offset;
    // todo: implement two methods described in 3.2 Iteration  Efficiency
    // 1. early termination
    // 2. summation truncation
    // the second one could improve speed significantly...

    // new patch pair
    updated = false;
    float& current_dist = min_distance_.at<float>(A_y, A_x);
    if (current_dist < 0.0f) {
      // first calculation for A(x, y)
      CalcDistance(*A_, A_x, A_y, *B_, B_x, B_y, half_patch_size_,
                   distance_type_, dist);
      current_dist = dist;
      updated = true;
    } else {
      bool ret = CalcDistance(*A_, A_x, A_y, *B_, B_x, B_y, half_patch_size_,
                              distance_type_, dist, current_dist);

      if (!ret) {
        // false when early termination happens
        return false;
      }

      if (ret && dist < current_dist) {
        current_dist = dist;
        updated = true;
      }
    }
    return true;
  }
};

bool Propagation(Image2f& nnf, int x, int y, DistanceCache& distance_cache);

bool RandomSearch(Image2f& nnf, int x, int y, int x_max, int y_max,
                  DistanceCache& distance_cache, float radius,
                  std::default_random_engine& engine,
                  std::uniform_real_distribution<float>& distribution_rs);

bool Initialize(Image2f& nnf, int B_w, int B_h, const Option& option,
                std::default_random_engine& engine);

inline float CalcDistance(const Image3b& A, const Image3b& B,
                          const Image2f& nnf, Image1f& distance, int x, int y,
                          const Option& option) {
  float dist;

  return dist;
}

inline bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B,
                         int B_x, int B_y, int half_patch_size,
                         DistanceType distance_type, float& distance) {
  if (distance_type == DistanceType::SSD) {
    return SSD(A, A_x, A_y, B, B_x, B_y, half_patch_size, distance);
  }

  return -9999.9f;
}

inline bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B,
                         int B_x, int B_y, int half_patch_size,
                         DistanceType distance_type, float& distance,
                         float current_min) {
  if (distance_type == DistanceType::SSD) {
    return SSD(A, A_x, A_y, B, B_x, B_y, half_patch_size, distance,
               current_min);
  }

  return -9999.9f;
}

inline bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                int B_y, int half_patch_size, float& val) {
  int& h_ps = half_patch_size;
  val = 0.0f;
  const float frac = 1.0f / 3.0f;
  for (int j = -h_ps; j < h_ps + 1; j++) {
    for (int i = -h_ps; i < h_ps + 1; i++) {
      auto& A_val = A.at<Vec3b>(A_y + j, A_x + i);
      auto& B_val = B.at<Vec3b>(B_y + j, B_x + i);
      std::array<float, 3> diff_list;

      for (int c = 0; c < 3; c++) {
        diff_list[c] = A_val[c] - B_val[c];
      }

      // average of 3 channels
      float diff = (diff_list[0] + diff_list[1] + diff_list[2]) * frac;
      val += (diff * diff);
    }
  }
  return true;
}

inline bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                int B_y, int half_patch_size, float& val, float current_min) {
  int& h_ps = half_patch_size;
  val = 0.0f;
  const float frac = 1.0f / 3.0f;
  for (int j = -h_ps; j < h_ps + 1; j++) {
    for (int i = -h_ps; i < h_ps + 1; i++) {
      const Vec3b& A_val = A.at<Vec3b>(A_y + j, A_x + i);
      const Vec3b& B_val = B.at<Vec3b>(B_y + j, B_x + i);
      std::array<float, 3> diff_list;

      for (int c = 0; c < 3; c++) {
        diff_list[c] = A_val[c] - B_val[c];
      }

      // average of 3 channels
      float diff = (diff_list[0] + diff_list[1] + diff_list[2]) * frac;
      val += (diff * diff);
      if (val > current_min) {
        return false;
      }
    }
  }
  return true;
}

#if 0
				inline float SSD(const Image1b& A, int A_x, int A_y, const Image1b& B, int B_x,
                 int B_y, int half_patch_size) {
  int& h_ps = half_patch_size;
  float ssd = 0.0f;
  for (int j = -h_ps; j < h_ps + 1; j++) {
    for (int i = -h_ps; i < h_ps + 1; i++) {
      unsigned char A_val = A.at<unsigned char>(A_y + j, A_x + i);
      unsigned char B_val = B.at<unsigned char>(B_y + j, B_x + i);
      float diff = A_val - B_val;
      ssd += (diff * diff);
    }
  }
  return ssd;
}
#endif  // 0

inline bool Propagation(Image2f& nnf, int x, int y, int x_max, int y_max,
                        DistanceCache& distance_cache) {
  bool updated{false};
  const int h_ps = distance_cache.half_patch_size();

  Vec2f& current_offset = nnf.at<Vec2f>(y, x);

  std::array<float, 3> dist_list;

  float current_dist = distance_cache.min_distance().at<float>(y, x);
  if (current_dist < 0.0f) {
    distance_cache.query(x, y, current_offset[0], current_offset[1],
                         current_dist, updated);
  }
  dist_list[0] = current_dist;

  Vec2f& offset_r = nnf.at<Vec2f>(y, x - 1);
  if (offset_r[0] + x - h_ps > 0 && offset_r[0] + x + h_ps < x_max &&
      offset_r[1] + y - h_ps > 0 && offset_r[1] + y + h_ps < y_max) {
    distance_cache.query(x, y, offset_r[0], offset_r[1], dist_list[1], updated);
  } else {
    dist_list[1] = std::numeric_limits<float>::max();
  }

  Vec2f& offset_u = nnf.at<Vec2f>(y - 1, x);
  if (offset_u[0] + x - h_ps > 0 && offset_u[0] + x + h_ps < x_max &&
      offset_u[1] + y - h_ps > 0 && offset_u[1] + y + h_ps < y_max) {
    distance_cache.query(x, y, offset_u[0], offset_u[1], dist_list[2], updated);
  } else {
    dist_list[2] = std::numeric_limits<float>::max();
  }

  auto& min_iter = std::min_element(dist_list.begin(), dist_list.end());
  size_t min_index = std::distance(dist_list.begin(), min_iter);

  if (min_index == 1) {
    current_offset = offset_r;
  } else if (min_index == 2) {
    current_offset = offset_u;
  }

  return true;
}

inline bool RandomSearch(
    Image2f& nnf, int x, int y, int x_max, int y_max,
    DistanceCache& distance_cache, float radius,
    std::default_random_engine& engine,
    std::uniform_real_distribution<float>& distribution_rs) {
  Vec2f& current = nnf.at<Vec2f>(y, x);
  int offset_x = static_cast<int>(distribution_rs(engine) * radius);
  int offset_y = static_cast<int>(distribution_rs(engine) * radius);
  Vec2f update = current;
  update[0] += offset_x;
  update[1] += offset_y;

  float h_ps = static_cast<float>(distance_cache.half_patch_size());
#if 0
				  update[0] =
      std::max(h_ps, std::min(update[0], static_cast<float>(x_max - h_ps - 1)));
  update[1] =
      std::max(h_ps, std::min(update[1], static_cast<float>(y_max - h_ps - 1)));

#endif  // 0
  if (update[0] + x < h_ps) {
    update[0] = h_ps - x;
  }
  if (update[0] + x > x_max - h_ps - 1) {
    update[0] = x_max - h_ps - 1 - x;
  }

  if (update[1] + y < h_ps) {
    update[1] = h_ps - y;
  }
  if (update[1] + y > y_max - h_ps - 1) {
    update[1] = y_max - h_ps - 1 - y;
  }

  float dist;
  bool updated{false};
  distance_cache.query(x, y, update[0], update[1], dist, updated);

  if (updated) {
    current = update;
    return true;
  }

  return false;
}

/* definition of interface */
inline bool Initialize(Image2f& nnf, int B_w, int B_h, const Option& option,
                       std::default_random_engine& engine) {
  if (option.init_type == InitType::RANDOM) {
    int h_ps = option.patch_size / 2;
    std::uniform_int_distribution<> dist_w(h_ps, B_w - 1 - h_ps);
    std::uniform_int_distribution<> dist_h(h_ps, B_h - 1 - h_ps);
    for (int j = h_ps; j < nnf.rows - h_ps; j++) {
      for (int i = h_ps; i < nnf.cols - h_ps; i++) {
        Vec2f& val = nnf.at<Vec2f>(j, i);
        val[0] = dist_w(engine) - i;
        val[1] = dist_h(engine) - j;
      }
    }
  } else {
    return false;
  }

  return true;
}

inline bool Compute(const Image3b& A, const Image3b& B, Image2f& nnf,
                    Image1f& distance, const Option& option) {
  std::default_random_engine engine(option.random_seed);
  std::uniform_real_distribution<float> distribution_rs(-1.0f, 1.0f);

  // memory allocation of nnf
  nnf = Image2f::zeros(A.rows, A.cols);

  // initialize
  Initialize(nnf, B.cols, B.rows, option, engine);
  DistanceCache distance_cache(A, B, option.distance_type,
                               option.patch_size / 2);

  // iteration
  for (int iter = 0; iter < option.max_iter; iter++) {
    printf("iter %d\n", iter);
    float radius = std::max(1.0f, option.w * std::pow(option.alpha, iter));
    const int h_ps = option.patch_size / 2;
    for (int j = h_ps; j < nnf.rows - h_ps; j++) {
      for (int i = h_ps; i < nnf.cols - h_ps; i++) {
        // Propagation
        Propagation(nnf, i, j, B.cols, B.rows, distance_cache);

        // Random search
        RandomSearch(nnf, i, j, B.cols, B.rows, distance_cache, radius, engine,
                     distribution_rs);
      }
    }
  }

  distance = Image1f::zeros(A.rows, A.cols);
  distance_cache.min_distance().copyTo(distance);

  return true;
}

inline bool ColorizeNnf(const Image2f& nnf, Image3b& vis_nnf,
                        float max_mag = 100.0f, float min_mag = 0.0f,
                        unsigned char v = 200) {
  // const float* data = reinterpret_cast<float*>(distance.data);
  // const int size = distance.cols * distance.rows;
  // const float max_d = *std::max_element(data, data + size);
  // const float min_d = *std::min_element(data, data + size);

  Image3b vis_nnf_hsv;
  vis_nnf_hsv = Image3b::zeros(nnf.rows, nnf.cols);

  float inv_2pi = 1.0f / (2 * M_PI);
  float inv_mag_factor = 1.0f / (max_mag - min_mag);
  for (int y = 0; y < vis_nnf_hsv.rows; y++) {
    for (int x = 0; x < vis_nnf_hsv.cols; x++) {
      const Vec2f& nn = nnf.at<Vec2f>(y, x);

      float angle = std::atan2(nn[1], nn[0]) + M_PI;
      float magnitude = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1]);
      //printf("angle %f\n", angle * inv_2pi * 360);
      //printf("magnitude %f\n", magnitude);
      float norm_magnitude = std::max(
          0.0f, std::min((magnitude - min_mag) * inv_mag_factor, 1.0f));

      Vec3b& hsv = vis_nnf_hsv.at<Vec3b>(y, x);
      hsv[0] = static_cast<unsigned char>(angle * inv_2pi * 179);
      hsv[1] = static_cast<unsigned char>(norm_magnitude * 255);
      hsv[2] = v;
    }
  }

  cv::cvtColor(vis_nnf_hsv, vis_nnf, cv::COLOR_HSV2BGR);

  return true;
}

#ifdef NANOPM_USE_TINYCOLORMAP
inline bool ColorizeDistance(const Image1f& distance, Image3b& vis_distance,
                             tinycolormap::ColormapType type) {
  const float* data = reinterpret_cast<float*>(distance.data);
  const int size = distance.cols * distance.rows;
  const float max_d = *std::max_element(data, data + size);
  const float min_d = *std::min_element(data, data + size);

  vis_distance = Image3b::zeros(distance.rows, distance.cols);

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_distance.rows; y++) {
    for (int x = 0; x < vis_distance.cols; x++) {
      const float& d = distance.at<float>(y, x);

      float norm_color = (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      const tinycolormap::Color& color =
          tinycolormap::GetColor(norm_color, type);

      Vec3b& vis = vis_distance.at<Vec3b>(y, x);
#ifdef NANOPM_USE_OPENCV
      // BGR
      vis[2] = static_cast<uint8_t>(color.r() * 255);
      vis[1] = static_cast<uint8_t>(color.g() * 255);
      vis[0] = static_cast<uint8_t>(color.b() * 255);
#else
      // RGB
      vis[0] = static_cast<uint8_t>(color.r() * 255);
      vis[1] = static_cast<uint8_t>(color.g() * 255);
      vis[2] = static_cast<uint8_t>(color.b() * 255);
#endif
    }
  }
}
#else
inline bool ColorizeDistance(const Image1f& distance, Image3b& vis_distance) {
  const float* data = reinterpret_cast<float*>(distance.data);
  const int size = distance.cols * distance.rows;
  const float max_d = *std::max_element(data, data + size);
  const float min_d = *std::min_element(data, data + size);

  vis_distance = Image3b::zeros(distance.rows, distance.cols);

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_distance.rows; y++) {
    for (int x = 0; x < vis_distance.cols; x++) {
      const float& d = distance.at<float>(y, x);

      float norm_color = (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      Vec3b& vis = vis_distance.at<Vec3b>(y, x);

      vis[2] = static_cast<uint8_t>(norm_color * 255);
      vis[1] = static_cast<uint8_t>(norm_color * 255);
      vis[0] = static_cast<uint8_t>(norm_color * 255);
    }
  }
  return true;
}
#endif

}  // namespace nanopm
