#pragma once
#include <vector>
#include <chrono>
#include <stdint.h>

#define CLAMP(v, lo, hi) ((v)>(hi) ? (hi) : ((v)<(lo)?(lo):(v)))
#define TO_U8(chan_fp) \
    (static_cast<uint8_t>(CLAMP((chan_fp)*255.0, 0.0, 255.0)))

struct alignas(sizeof(double)) Color {
  double r, g, b, a;
};

std::vector<Color> LoadImage(const char *path, size_t &width, size_t &height);

void SaveImage(const std::vector<Color> &image, size_t width, size_t height,
               const char *path);

struct Config {
  struct {
    double scale;
  } saturation;
  struct {
    int8_t brightness, contrast;
  } brightnessContrast;
  struct {
    float blackLevel, exposure;
  } exposure;
};

Config LoadConfig(const char *path);

struct TimeVals {
  float mean, Mse, perfDenom;
};

void CalcTimeVals(TimeVals &res,
                  std::vector<std::chrono::milliseconds> &timings,
                  const size_t nSkip);
void CalcTimeVals(TimeVals &res, std::vector<float> &timings,
                  const size_t nSkip);
