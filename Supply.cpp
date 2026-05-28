#include "Supply.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <nlohmann/json.hpp>

using namespace std;

vector<Color> LoadImage(const char *path, size_t &width, size_t &height) {
  int iwidth{}, iheight{}, ichannels{};
  uint8_t *data = stbi_load(path, &iwidth, &iheight, &ichannels, 4);

  if (!data) {
    std::ostringstream oss;
    oss << "Failed to load '" << path << "'";
    throw std::runtime_error(oss.str());
  }

  assert(iwidth != 0);
  assert(iheight != 0);
  assert(ichannels == 4);

  size_t size = iwidth * iheight;
  vector<Color> image(size);
  for (size_t i = 0; i < size; i++) {
    image[i].r = data[i * 4 + 0] / 255.0;
    image[i].g = data[i * 4 + 1] / 255.0;
    image[i].b = data[i * 4 + 2] / 255.0;
    image[i].a = data[i * 4 + 3] / 255.0;
  }

  stbi_image_free(data);
  width  = iwidth;
  height = iheight;
  return image;
}

void SaveImage(const vector<Color> &image, size_t width, size_t height,
               const char *path) {
  assert(!image.empty());
  assert(image.size() == width * height);

  vector<uint8_t> buffer(image.size() * 4);
  for (size_t i = 0; i < width * height; i++) {
    buffer[i*4+0]=TO_U8(image[i].r);
    buffer[i*4+1]=TO_U8(image[i].g);
    buffer[i*4+2]=TO_U8(image[i].b);
    buffer[i*4+3]=TO_U8(image[i].a);
  }

  stbi_write_png(path, width, height, 4, buffer.data(),
                 width * 4 * sizeof(uint8_t));
}

Config LoadConfig(const char *path) {
  ifstream configFile(path);
  nlohmann::json configJson;
  configFile>>configJson;
  if (!configFile)
    throw std::runtime_error("Failed to parse JSON configuration file");
  
  Config result;
  result.saturation.scale=(double)configJson["Saturation"]["scale"];

  result.brightnessContrast.brightness=
      (int8_t)configJson["BrightnessContrast"]["brightness"];
  result.brightnessContrast.contrast=
      (int8_t)configJson["BrightnessContrast"]["contrast"];

  result.exposure.blackLevel=(float)configJson["Exposure"]["black_level"];
  result.exposure.exposure=(float)configJson["Exposure"]["exposure"];
  return result;
}

void CalcTimeVals(TimeVals &res, vector<chrono::milliseconds> &timings,
                     const size_t nSkip) {
  res=TimeVals{0.0f, 0.0f, 0.0f};

  sort(timings.begin(), timings.end());
  for (unsigned i=nSkip; i<timings.size()-nSkip; ++i)
    res.mean+=timings[i].count();
  res.mean/=timings.size()-2*nSkip;

  for (unsigned i=nSkip; i<timings.size()-nSkip; ++i)
    res.Mse+=(timings[i].count()-res.mean)*(timings[i].count()-res.mean);
  res.Mse=sqrt(res.Mse/(timings.size()-2*nSkip));

  res.perfDenom=res.mean*res.mean-res.Mse*res.Mse;
}

void CalcTimeVals(TimeVals &res, vector<float> &timings, const size_t nSkip) {
  res=TimeVals{0.0f, 0.0f, 0.0f};

  sort(timings.begin(), timings.end());
  for (unsigned i=nSkip; i<timings.size()-nSkip; ++i)
    res.mean+=timings[i];
  res.mean/=timings.size()-2*nSkip;

  for (unsigned i=nSkip; i<timings.size()-nSkip; ++i)
    res.Mse+=(timings[i]-res.mean)*(timings[i]-res.mean);
  res.Mse=sqrt(res.Mse/(timings.size()-2*nSkip));

  res.perfDenom=res.mean*res.mean-res.Mse*res.Mse;
}