/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 *
 * NanoPM, single header only PatchMatch
 *
 * Reference
 *  - Barnes, Connelly, et al. "PatchMatch: A randomized correspondence
 * algorithm for structural image editing." ACM Transactions on Graphics (ToG).
 * Vol. 28. No. 3. ACM, 2009.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <array>
#include <random>
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

/* declation of public interface */

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

  float w = 16.0f;
  float alpha = 0.5f;

  InitType init_type = InitType::RANDOM;
  Image2f initial;
  int initial_random_iter = 5;

  unsigned int random_seed = 0;  // for repeatability

  DistanceType distance_type = DistanceType::SSD;

  bool verbose = true;
  std::string debug_dir = "";
};

bool Compute(const Image3b& A, const Image3b& B, Image2f& nnf,
             Image1f& distance, const Option& option);

// 3 channel only
// float* nnf and float* distance must be allocated by caller
bool Compute(const unsigned char* A, int A_w, int A_h, const unsigned char* B,
             int B_w, int B_h, float* nnf, float* distance,
             const Option& option);

// brute force method for reference as Ground Truth
bool BruteForce(const Image3b& A, const Image3b& B, Image2f& nnf,
                Image1f& distance, const Option& option);

bool ColorizeNnf(const Image2f& nnf, Image3b& vis_nnf, float max_mag = 100.0f,
                 float min_mag = 0.0f, unsigned char v = 255);

#ifdef NANOPM_USE_TINYCOLORMAP
bool ColorizeDistance(const Image1f& distance, Image3b& vis_distance,
                      tinycolormap::ColormapType type);
#endif

bool ColorizeDistance(const Image1f& distance, Image3b& vis_distance);

/* end of declation of interface */

namespace impl {
/* declation of private interface */
bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                  int B_y, int patch_size_x, int patch_size_y,
                  DistanceType distance_type, float& distance);

bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                  int B_y, int patch_size_x, int patch_size_y,
                  DistanceType distance_type, float& distance,
                  float current_min);

bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x, int B_y,
         int patch_size_x, int patch_size_y, float& val);

bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x, int B_y,
         int patch_size_x, int patch_size_y, float& val, float current_min);

class DistanceCache {
  const Image3b* A_;
  const Image3b* B_;
  int patch_size_;
  Image1f min_distance_;
  DistanceType distance_type_;

 public:
  DistanceCache() = delete;
  DistanceCache(const DistanceCache& src) = delete;
  ~DistanceCache() = default;
  DistanceCache(const Image3b& A, const Image3b& B,
                const DistanceType& distance_type, int patch_size)
      : A_(&A), B_(&B), distance_type_(distance_type), patch_size_(patch_size) {
    min_distance_ = Image1f::zeros(A_->rows, A_->cols);
    min_distance_.setTo(-1.0f);
  }

  Image1f& min_distance() { return min_distance_; }
  int patch_size() const { return patch_size_; }
  const Image3b* A() const { return A_; }
  const Image3b* B() const { return B_; }
  DistanceType distance_type() const { return distance_type_; }
  bool query(int A_x, int A_y, int x_offset, int y_offset, float& dist,
             bool& updated) {
    int B_x = A_x + x_offset;
    int B_y = A_y + y_offset;

    // new patch pair
    updated = false;
    float& current_dist = min_distance_.at<float>(A_y, A_x);

#if 0
    CalcDistance(*A_, A_x, A_y, *B_, B_x, B_y, patch_size_, patch_size_,
                 distance_type_, dist);
    if (dist < current_dist) {
      current_dist = dist;
      updated = true;
    }

#else
    // with early termination
    // todo: maybe slow by internal if

    if (current_dist < 0.0f) {
      // first calculation for A(x, y)
      CalcDistance(*A_, A_x, A_y, *B_, B_x, B_y, patch_size_, patch_size_,
                   distance_type_, dist);
      current_dist = dist;
      updated = true;
    } else {
      // early termination version
      bool ret = CalcDistance(*A_, A_x, A_y, *B_, B_x, B_y, patch_size_,
                              patch_size_, distance_type_, dist, current_dist);

      if (!ret) {
        // false when early termination happens
        return false;
      }

      if (ret && dist < current_dist) {
        current_dist = dist;
        updated = true;
      }
    }
#endif  // 0

    return true;
  }
};

bool Propagation(Image2f& nnf, int x, int y, int x_max, int y_max,
                 DistanceCache& distance_cache);

bool RandomSearch(Image2f& nnf, int x, int y, int x_max, int y_max,
                  DistanceCache& distance_cache, float radius,
                  std::default_random_engine& engine,
                  std::uniform_real_distribution<float>& distribution_rs);

bool Initialize(Image2f& nnf, int B_w, int B_h, const Option& option,
                std::default_random_engine& engine);

bool UpdateOffsetWithGuard(Vec2f& offset, int patch_size, int x, int y,
                           int x_max, int y_max);

bool DebugDump(const std::string& debug_dir, const std::string& postfix,
               const Image2f& nnf, const Image1f& distance);

/* end of declation of private interface */
}  // namespace impl

/* definition of interface */
inline bool Compute(const unsigned char* A, int A_w, int A_h,
                    const unsigned char* B, int B_w, int B_h, float* nnf,
                    float* distance, const Option& option) {
  Image2f nnf_;
  Image1f distance_;

  Image3b A_, B_;
  A_ = Image3b::zeros(A_h, A_w);
  std::memcpy(A_.data, A, sizeof(unsigned char) * 3 * A_w * A_h);

  B_ = Image3b::zeros(B_h, B_w);
  std::memcpy(B_.data, B, sizeof(unsigned char) * 3 * B_w * B_h);

  bool ret = Compute(A_, B_, nnf_, distance_, option);

  std::memcpy(nnf, reinterpret_cast<float*>(nnf_.data),
              sizeof(float) * nnf_.cols * nnf_.rows);
  std::memcpy(distance, reinterpret_cast<float*>(distance_.data),
              sizeof(float) * distance_.cols * distance_.rows);
  return ret;
}

inline bool Compute(const Image3b& A, const Image3b& B, Image2f& nnf,
                    Image1f& distance, const Option& option) {
  std::default_random_engine engine(option.random_seed);
  std::uniform_real_distribution<float> distribution_rs(-1.0f, 1.0f);

  // memory allocation of nnf
  nnf = Image2f::zeros(A.rows, A.cols);

  // initialize
  impl::Initialize(nnf, B.cols, B.rows, option, engine);
  // return true;
  impl::DistanceCache distance_cache(A, B, option.distance_type,
                                     option.patch_size);

  // iteration
  for (int iter = 0; iter < option.max_iter; iter++) {
    float radius = std::max(1.0f, option.w * std::pow(option.alpha, iter));
    if (option.verbose) {
      printf("iter %d radious %f \n", iter, radius);
    }
    // todo: paralellize here.
    // "in practice long propagations are not needed"
    // See "3.2 Iteration GPU implementation." of the original paper
    for (int j = 0; j < nnf.rows - option.patch_size; j++) {
      if (j % (nnf.rows / 4) == 0 && !option.debug_dir.empty()) {
        impl::DebugDump(
            option.debug_dir,
            std::to_string(iter) + "_" + std::to_string(j / (nnf.rows / 4)),
            nnf, distance_cache.min_distance());
      }

      for (int i = 0; i < nnf.cols - option.patch_size; i++) {
        // Propagation
        impl::Propagation(nnf, i, j, B.cols, B.rows, distance_cache);

        // Random search
        impl::RandomSearch(nnf, i, j, B.cols, B.rows, distance_cache, radius,
                           engine, distribution_rs);
      }
    }

    if (!option.debug_dir.empty()) {
      impl::DebugDump(option.debug_dir, std::to_string(iter) + "_4", nnf,
                      distance_cache.min_distance());
    }
  }

  distance = Image1f::zeros(A.rows, A.cols);
  distance_cache.min_distance().copyTo(distance);

  return true;
}

inline bool BruteForce(const Image3b& A, const Image3b& B, Image2f& nnf,
                       Image1f& distance, const Option& option) {
  // memory allocation
  nnf = Image2f::zeros(A.rows, A.cols);
  distance = Image1f::zeros(A.rows, A.cols);
  distance.setTo(std::numeric_limits<float>::max());
#ifdef NANOPM_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < nnf.rows - option.patch_size; j++) {
    if (option.verbose && j % 10 == 0) {
      printf("current row %d\n", j);
    }

    for (int i = 0; i < nnf.cols - option.patch_size; i++) {
      // iterate all patches in B
      float& current_dist = distance.at<float>(j, i);
      Vec2f& current_nn = nnf.at<Vec2f>(j, i);
      for (int jj = 0; jj < B.rows - option.patch_size; jj++) {
        for (int ii = 0; ii < B.cols - option.patch_size; ii++) {
          float dist;
          impl::CalcDistance(A, i, j, B, ii, jj, option.patch_size,
                             option.patch_size, option.distance_type, dist);

          if (dist < current_dist) {
            current_dist = dist;
            current_nn[0] = ii - i;
            current_nn[1] = jj - j;
          }
        }
      }
    }
  }

  return true;
}

inline bool ColorizeNnf(const Image2f& nnf, Image3b& vis_nnf, float max_mag,
                        float min_mag, unsigned char v) {
  Image3b vis_nnf_hsv;
  vis_nnf_hsv = Image3b::zeros(nnf.rows, nnf.cols);

  float inv_2pi = static_cast<float>(1.0f / (2 * M_PI));
  float inv_mag_factor = 1.0f / (max_mag - min_mag);
  for (int y = 0; y < vis_nnf_hsv.rows; y++) {
    for (int x = 0; x < vis_nnf_hsv.cols; x++) {
      const Vec2f& nn = nnf.at<Vec2f>(y, x);

      float angle = static_cast<float>(std::atan2(nn[1], nn[0]) + M_PI);
      float magnitude = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1]);
      // printf("angle %f\n", angle * inv_2pi * 360);
      // printf("magnitude %f\n", magnitude);
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
#endif
inline bool ColorizeDistance(const Image1f& distance, Image3b& vis_distance) {
  const float* raw_data = reinterpret_cast<float*>(distance.data);
  const int size = distance.cols * distance.rows;
  std::vector<float> valid_data;
  for (int i = 0; i < size; i++) {
    if (raw_data[i] > 0.0f) {
      valid_data.push_back(raw_data[i]);
    }
  }
  std::sort(valid_data.begin(), valid_data.end());
  float r = 0.05f;
  float min_d = -1.0f;
  float max_d = 0.0f;
  if (!valid_data.empty()) {
    // get 5% an 95% percentile...
    min_d = valid_data[static_cast<size_t>(valid_data.size() * r)];
    max_d = valid_data[static_cast<size_t>(valid_data.size() * (1.0f - r))];
  }

  vis_distance = Image3b::zeros(distance.rows, distance.cols);

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_distance.rows; y++) {
    for (int x = 0; x < vis_distance.cols; x++) {
      const float& d = distance.at<float>(y, x);

      float norm_color = d < 0.0f ? 1.0f : (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      Vec3b& vis = vis_distance.at<Vec3b>(y, x);

      vis[2] = static_cast<uint8_t>(norm_color * 255);
      vis[1] = static_cast<uint8_t>(norm_color * 255);
      vis[0] = static_cast<uint8_t>(norm_color * 255);
    }
  }
  return true;
}
/* end of definition of interface */

namespace impl {

/* definition of interface */
inline bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B,
                         int B_x, int B_y, int patch_size_x, int patch_size_y,
                         DistanceType distance_type, float& distance) {
  if (distance_type == DistanceType::SSD) {
    return SSD(A, A_x, A_y, B, B_x, B_y, patch_size_x, patch_size_y, distance);
  }

  return false;
}

inline bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B,
                         int B_x, int B_y, int patch_size_x, int patch_size_y,
                         DistanceType distance_type, float& distance,
                         float current_min) {
  if (distance_type == DistanceType::SSD) {
    return SSD(A, A_x, A_y, B, B_x, B_y, patch_size_x, patch_size_y, distance,
               current_min);
  }

  return false;
}

inline bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                int B_y, int patch_size_x, int patch_size_y, float& val) {
  val = 0.0f;
  const float frac = 1.0f / 3.0f;
  for (int j = 0; j < patch_size_y; j++) {
    for (int i = 0; i < patch_size_x; i++) {
      auto& A_val = A.at<Vec3b>(A_y + j, A_x + i);
      auto& B_val = B.at<Vec3b>(B_y + j, B_x + i);
      std::array<float, 3> diff_list;

      for (int c = 0; c < 3; c++) {
        diff_list[c] = static_cast<float>(A_val[c] - B_val[c]);
      }

      // average of 3 channels
      float diff = (diff_list[0] + diff_list[1] + diff_list[2]) * frac;
      val += (diff * diff);
    }
  }
  return true;
}

inline bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                int B_y, int patch_size_x, int patch_size_y, float& val,
                float current_min) {
  val = 0.0f;
  const float frac = 1.0f / 3.0f;
  for (int j = 0; j < patch_size_y; j++) {
    for (int i = 0; i < patch_size_x; i++) {
      const Vec3b& A_val = A.at<Vec3b>(A_y + j, A_x + i);
      const Vec3b& B_val = B.at<Vec3b>(B_y + j, B_x + i);
      std::array<float, 3> diff_list;

      for (int c = 0; c < 3; c++) {
        diff_list[c] = static_cast<float>(A_val[c] - B_val[c]);
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

inline bool UpdateOffsetWithGuard(Vec2f& offset, int patch_size, int x, int y,
                                  int x_max, int y_max) {
  bool ret{false};
  float new_x = offset[0] + x;
  if (new_x < 0) {
    offset[0] = static_cast<float>(-x);
    ret = true;
  } else if (new_x > x_max - 1 - patch_size) {
    offset[0] = static_cast<float>(x_max - 1 - patch_size - x);
    ret = true;
  }

  float new_y = offset[1] + y;
  if (new_y < 0) {
    offset[1] = static_cast<float>(-y);
    ret = true;
  } else if (new_y > y_max - 1 - patch_size) {
    offset[1] = static_cast<float>(y_max - 1 - patch_size - y);
    ret = true;
  }

  return ret;
}

inline bool Propagation(Image2f& nnf, int x, int y, int x_max, int y_max,
                        DistanceCache& distance_cache) {
  bool updated{false};

  Vec2f& current_offset = nnf.at<Vec2f>(y, x);

  std::array<float, 3> dist_list;

  float& current_dist = distance_cache.min_distance().at<float>(y, x);
  if (current_dist < 0.0f) {
    distance_cache.query(x, y, static_cast<int>(current_offset[0]),
                         static_cast<int>(current_offset[1]), current_dist,
                         updated);
  }
  dist_list[0] = current_dist;

  Vec2f offset_l;
  if (x > 0) {
    offset_l = nnf.at<Vec2f>(y, x - 1);
    bool gurded = UpdateOffsetWithGuard(offset_l, distance_cache.patch_size(),
                                        x, y, x_max, y_max);

    float& current_l_dist = distance_cache.min_distance().at<float>(y, x - 1);

    if (gurded || current_l_dist < 0.0f) {
      distance_cache.query(x, y, static_cast<int>(offset_l[0]),
                           static_cast<int>(offset_l[1]), dist_list[1],
                           updated);
    } else {
      // integral image like technique described in "3.2 Iteration
      // Efficiency."

      // substract left most col
      float l_dist;
      CalcDistance(*distance_cache.A(), x - 1, y, *distance_cache.B(),
                   static_cast<int>(offset_l[0]) + x - 1,
                   static_cast<int>(offset_l[1]) + y, 1,
                   distance_cache.patch_size(), distance_cache.distance_type(),
                   l_dist);

      // add right col
      float r_dist;
      CalcDistance(
          *distance_cache.A(), x + distance_cache.patch_size() - 1, y,
          *distance_cache.B(),
          static_cast<int>(offset_l[0]) + x + distance_cache.patch_size() - 1,
          static_cast<int>(offset_l[1]) + y, 1, distance_cache.patch_size(),
          distance_cache.distance_type(), r_dist);

      dist_list[1] = current_l_dist - l_dist + r_dist;
    }

  } else {
    dist_list[1] = std::numeric_limits<float>::max();
  }

  Vec2f offset_u;
  if (y > 0) {
    offset_u = nnf.at<Vec2f>(y - 1, x);
    bool gurded = UpdateOffsetWithGuard(offset_u, distance_cache.patch_size(),
                                        x, y, x_max, y_max);

    float& current_u_dist = distance_cache.min_distance().at<float>(y - 1, x);

    if (gurded || current_u_dist < 0.0f) {
      distance_cache.query(x, y, static_cast<int>(offset_u[0]),
                           static_cast<int>(offset_u[1]), dist_list[2],
                           updated);

    } else {
      // integral image like technique described in "3.2 Iteration
      // Efficiency."

      // substract upper most col
      float u_dist;
      CalcDistance(*distance_cache.A(), x, y - 1, *distance_cache.B(),
                   static_cast<int>(offset_u[0]) + x,
                   static_cast<int>(offset_u[1]) + y - 1,
                   distance_cache.patch_size(), 1,
                   distance_cache.distance_type(), u_dist);

      // add bottom col
      float b_dist;
      CalcDistance(
          *distance_cache.A(), x, y + distance_cache.patch_size() - 1,
          *distance_cache.B(), static_cast<int>(offset_u[0]) + x,
          static_cast<int>(offset_u[1]) + y + distance_cache.patch_size() - 1,
          distance_cache.patch_size(), 1, distance_cache.distance_type(),
          b_dist);

      dist_list[2] = current_u_dist - u_dist + b_dist;
    }
  } else {
    dist_list[2] = std::numeric_limits<float>::max();
  }

  auto min_iter = std::min_element(dist_list.begin(), dist_list.end());
  size_t min_index = std::distance(dist_list.begin(), min_iter);

  if (min_index == 1) {
    current_offset = offset_l;
    current_dist = dist_list[1];

  } else if (min_index == 2) {
    current_offset = offset_u;
    current_dist = dist_list[2];
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

  UpdateOffsetWithGuard(update, distance_cache.patch_size(), x, y, x_max,
                        y_max);

  float dist;
  bool updated{false};
  distance_cache.query(x, y, static_cast<int>(update[0]),
                       static_cast<int>(update[1]), dist, updated);

  if (updated) {
    current = update;
    return true;
  }

  return false;
}

inline bool Initialize(Image2f& nnf, int B_w, int B_h, const Option& option,
                       std::default_random_engine& engine) {
  if (option.init_type == InitType::RANDOM) {
    std::uniform_int_distribution<> dist_w(0, B_w - 1 - option.patch_size);
    std::uniform_int_distribution<> dist_h(0, B_h - 1 - option.patch_size);
    for (int j = 0; j < nnf.rows - option.patch_size; j++) {
      for (int i = 0; i < nnf.cols - option.patch_size; i++) {
        Vec2f& val = nnf.at<Vec2f>(j, i);
        val[0] = static_cast<float>(dist_w(engine) - i);
        val[1] = static_cast<float>(dist_h(engine) - j);
      }
    }
  } else if (option.init_type == InitType::INITIAL) {
    option.initial.copyTo(nnf);
  } else {
    return false;
  }

  return true;
}

inline bool DebugDump(const std::string& debug_dir, const std::string& postfix,
                      const Image2f& nnf, const Image1f& distance) {
  Image3b vis_nnf, vis_distance;
  nanopm::ColorizeNnf(nnf, vis_nnf);
  nanopm::imwrite(debug_dir + "nnf_" + postfix + ".jpg", vis_nnf);
  nanopm::ColorizeDistance(distance, vis_distance);
  nanopm::imwrite(debug_dir + "distance_" + postfix + ".jpg", vis_distance);

  return true;
}

}  // namespace impl
}  // namespace nanopm
