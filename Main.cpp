#include "Supply.h"
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include "shaders/filtersShaderIR.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#define _USE_MATH_DEFINES
#include <math.h>

#define N_WARMUP (10)
#define N_ITERS (100)
#define N_SKIP (5)

#define AlignDown(offset, alignment) \
  ((offset)-(offset)%(alignment))

#define AlignUp(offset, alignment) \
  (((offset)+(alignment)-1)-((offset)+(alignment)-1)%(alignment))

#define WORKGROUP_SIZE (256)

using namespace std;

// Cpu version:
void Saturation(vector<Color> &dst, const vector<Color> &src, double scale) {
  assert(!src.empty());
  assert(dst.size() == src.size());
  assert(scale >= 0.0);
  assert(scale <= 2.0);

  double rscale = 1.0 - scale;
  for (size_t i = 0; i < src.size(); i++)
  {
    double desaturated = (src[i].r * 0.2126 +
                          src[i].g * 0.7152 +
                          src[i].b * 0.0722) * rscale;
    dst[i].r = desaturated + src[i].r * scale;
    dst[i].g = desaturated + src[i].g * scale;
    dst[i].b = desaturated + src[i].b * scale;
    dst[i].a = src[i].a;
  }
}

void BrightnessContrast(vector<Color> &dst, const vector<Color> &src,
                        int8_t brightness, int8_t contrast) {
  assert(!src.empty());
  assert(dst.size() == src.size());
  assert(abs(brightness) <= 127);
  assert(abs(contrast)   <= 127);

  double brightness_shift = brightness / 127.0;
  double contrast_factor = tan((contrast / 127.0 + 1.0) * (M_PI / 4.0));
  for (size_t i = 0; i < src.size(); i++)
  {
    dst[i].r = (src[i].r - 0.5) * contrast_factor + brightness_shift + 0.5;
    dst[i].g = (src[i].g - 0.5) * contrast_factor + brightness_shift + 0.5;
    dst[i].b = (src[i].b - 0.5) * contrast_factor + brightness_shift + 0.5;
    dst[i].a = src[i].a;
  }
}

void Exposure(vector<Color> &dst, const vector<Color> &src,
              float black_level, float exposure) {
  assert(!src.empty());
  assert(dst.size() == src.size());
  assert(abs(black_level) <= 0.1);
  assert(abs(exposure)    <= 10.0);

  double white = exp(-exposure);
  double diff  = max(white - black_level, 0.000001);
  double gain  = 1.0f / diff;

  for (size_t i = 0;  i <src.size(); i++)
  {
    dst[i].r = (src[i].r - black_level) * gain;
    dst[i].g = (src[i].g - black_level) * gain;
    dst[i].b = (src[i].b - black_level) * gain;
    dst[i].a = src[i].a;
  }
}

vector<Color> ProcessImageCpu(const vector<Color> &img, const Config config,
                              chrono::milliseconds *elapsedTime) {
  static vector<Color> buf(img.size());
  vector<Color> result(img.size());
  
  auto startTime=chrono::high_resolution_clock::now();
  Saturation(result, img, config.saturation.scale);
  BrightnessContrast(buf, result, config.brightnessContrast.brightness,
                                  config.brightnessContrast.contrast);
  Exposure(result, buf, config.exposure.blackLevel,
                        config.exposure.exposure);
  auto endTime=chrono::high_resolution_clock::now();

  if (elapsedTime)
    *elapsedTime=chrono::duration_cast<chrono::milliseconds>(endTime-startTime);
  
  return result;
}

struct {
  VkInstance instance;
  VkPhysicalDevice physDevice;
  VkDevice device;
  VkQueue queue;
  VmaAllocator allocator;
  uint64_t timestampMask;
  uint32_t queueFamilyIdx;
} gVulkanContext;

VkShaderModule CreateShaderModuleFromIR(const uint32_t *shaderIR,
                                        const uint32_t IRSize) {
  VkShaderModuleCreateInfo shaderModuleCI={};
  shaderModuleCI.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCI.codeSize=IRSize;
  shaderModuleCI.pCode=shaderIR;
  
  VkShaderModule result;
  vkCreateShaderModule(gVulkanContext.device, &shaderModuleCI, NULL, &result);
  return result;
}

VkShaderModule CreateShaderModuleFromFile(const char shaderFilename[]) {
  FILE *shaderFile=fopen(shaderFilename, "rb");
  assert(shaderFile);
  
  fseek(shaderFile, 0, SEEK_END);
  long shaderFileSize=ftell(shaderFile);
  assert(shaderFileSize!=-1);
  fseek(shaderFile, 0, SEEK_SET);

  vector<uint32_t> shaderCode((shaderFileSize+sizeof(uint32_t)-1)/
                              sizeof(uint32_t));
  size_t bytesRead=fread(shaderCode.data(), 1, shaderFileSize, shaderFile);
  assert(bytesRead==static_cast<size_t>(shaderFileSize));
  
  fclose(shaderFile);

  return CreateShaderModuleFromIR(shaderCode.data(), AlignUp(bytesRead, 4));
}

void InitializeVulkan() {
  {
  /// Specifying Application Info
  VkApplicationInfo applicationI={};
  applicationI.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationI.pApplicationName="vulkan-filters";
  applicationI.applicationVersion=1;
  applicationI.apiVersion=VK_API_VERSION_1_0;
  
  /// Creating the Instance itself
  const char **layers=NULL;
  VkInstanceCreateInfo instanceCI={};
  instanceCI.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCI.pApplicationInfo=&applicationI;
  instanceCI.enabledLayerCount=0;
  instanceCI.ppEnabledLayerNames=layers;
  instanceCI.enabledExtensionCount=0;
  instanceCI.ppEnabledExtensionNames=NULL;
  
  if (vkCreateInstance(&instanceCI, NULL, &gVulkanContext.instance)!=VK_SUCCESS)
  {
    fprintf(stderr, "Error: can't create Vulkan instance.\n");
    exit(-1);
  }
  }

  /// Searching for suitable physical device and queue family
  {
  uint32_t availablePhysDevicesCount=0;
  vkEnumeratePhysicalDevices(gVulkanContext.instance,
                             &availablePhysDevicesCount, NULL);
  vector<VkPhysicalDevice> availablePhysDevices(availablePhysDevicesCount);
  vkEnumeratePhysicalDevices(gVulkanContext.instance,
                             &availablePhysDevicesCount,
                             availablePhysDevices.data());
  
  for (uint32_t PDI=0; PDI<availablePhysDevicesCount; ++PDI) {
    auto checkPhysDevice4shaderFloat64=[](const VkPhysicalDevice physDevice)
        -> bool {
      VkPhysicalDeviceFeatures features;
      vkGetPhysicalDeviceFeatures(physDevice, &features);

      return features.shaderFloat64;
    };

    if (!checkPhysDevice4shaderFloat64(availablePhysDevices[PDI]))
      continue;
    
    uint32_t availableQueueFamiliesCount=0;
    vkGetPhysicalDeviceQueueFamilyProperties(availablePhysDevices[PDI],
                                             &availableQueueFamiliesCount,
                                             NULL);
    vector<VkQueueFamilyProperties> availableQueueFamilies(
                                        availableQueueFamiliesCount);
    vkGetPhysicalDeviceQueueFamilyProperties(availablePhysDevices[PDI],
                                             &availableQueueFamiliesCount,
                                             availableQueueFamilies.data());
    
    for (uint32_t QFI=0; QFI<availableQueueFamiliesCount; ++QFI)
      if (availableQueueFamilies[QFI].queueFlags & VK_QUEUE_COMPUTE_BIT &&
          availableQueueFamilies[QFI].timestampValidBits) {
        gVulkanContext.physDevice=availablePhysDevices[PDI];
        gVulkanContext.queueFamilyIdx=QFI;
        gVulkanContext.timestampMask=
            ~(uint64_t)0>>64-availableQueueFamilies[QFI].timestampValidBits;
        goto device8queue_found;
      }
  }
  fprintf(stderr, "Error: can't find suitable physical device.\n");
  vkDestroyInstance(gVulkanContext.instance, NULL);
  exit(-1);
  
device8queue_found:
  ;
  }

  /// Creating logical device
  {
  constexpr float queuePriority=1.0f;
  /// Specifying one family Queues Create Info
  VkDeviceQueueCreateInfo queueCI={};
  queueCI.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueFamilyIndex=gVulkanContext.queueFamilyIdx;
  queueCI.queueCount=1;
  queueCI.pQueuePriorities=&queuePriority;

  /// Creating Logical Device with desired compute-capable queue
  VkPhysicalDeviceFeatures features={};
  features.shaderFloat64=VK_TRUE;
  VkDeviceCreateInfo deviceCI={};
  deviceCI.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCI.queueCreateInfoCount=1;
  deviceCI.pQueueCreateInfos=&queueCI;
  deviceCI.pEnabledFeatures=&features;
  
  vkCreateDevice(gVulkanContext.physDevice, &deviceCI, NULL,
                 &gVulkanContext.device);
  vkGetDeviceQueue(gVulkanContext.device, gVulkanContext.queueFamilyIdx, 0,
                   &gVulkanContext.queue);

  VmaAllocatorCreateInfo allocatorCI={};
  allocatorCI.flags=VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
  allocatorCI.physicalDevice=gVulkanContext.physDevice;
  allocatorCI.device=gVulkanContext.device;
  allocatorCI.instance=gVulkanContext.instance;
  if (vmaCreateAllocator(&allocatorCI, &gVulkanContext.allocator)!=VK_SUCCESS) {
    fprintf(stderr, "Error: cant't initialize VMA library.\n");
    vkDestroyDevice(gVulkanContext.device, NULL);
    vkDestroyInstance(gVulkanContext.instance, NULL);
    exit(-1);
  }
  }
}

void FinalizeVulkan() {
  vmaDestroyAllocator(gVulkanContext.allocator);
  vkDestroyDevice(gVulkanContext.device, NULL);
  vkDestroyInstance(gVulkanContext.instance, NULL);
}

struct {
  VkCommandPool cmdPool;
  VkCommandBuffer filtersCmdBuf;
  VkPipeline filtersPipeline;
  VkDescriptorPool descPool;
  VkFence fence;
  VkQueryPool queryPool;
  struct {
    VkBuffer buf;
    VmaAllocation alloc;
  } imgBuffer, configBuffer;
  VkSubmitInfo submitInfo;
} gProcessingContext;

void InitializeProcessingContext(const uint32_t imgSize) {
  // Creating buffers
  {
  VkBufferCreateInfo bufferCI={};
  bufferCI.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCI.size=imgSize*sizeof(Color);
  bufferCI.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferCI.sharingMode=VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocInfo={};
  allocInfo.flags=VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  allocInfo.usage=VMA_MEMORY_USAGE_AUTO;
  if (vmaCreateBuffer(gVulkanContext.allocator, &bufferCI, &allocInfo,
                      &gProcessingContext.imgBuffer.buf,
                      &gProcessingContext.imgBuffer.alloc,
                      NULL)
      !=VK_SUCCESS) {
    fprintf(stderr,
            "Error: can't allocate device memroy for image.\n");
    FinalizeVulkan();
    exit(-1);
  }

  bufferCI.size=4*sizeof(double)+sizeof(float);
  bufferCI.usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (vmaCreateBuffer(gVulkanContext.allocator, &bufferCI, &allocInfo,
                      &gProcessingContext.configBuffer.buf,
                      &gProcessingContext.configBuffer.alloc, NULL)
      !=VK_SUCCESS) {
    fprintf(stderr,
            "Error: can't allocate device memroy for config buffer.\n");
    vmaDestroyBuffer(gVulkanContext.allocator, gProcessingContext.imgBuffer.buf,
                     gProcessingContext.imgBuffer.alloc);
    FinalizeVulkan();
    exit(-1);
  }
  }

  // Creating layouts of descriptor sets
  VkDescriptorSetLayout descSetLayout;
  {
  VkDescriptorSetLayoutBinding descBindings[2]={};
  descBindings[0].binding=0;
  descBindings[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descBindings[0].descriptorCount=1;
  descBindings[0].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;

  descBindings[1].binding=1;
  descBindings[1].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descBindings[1].descriptorCount=1;
  descBindings[1].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo descSetLayoutCI={};
  descSetLayoutCI.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descSetLayoutCI.bindingCount=2;
  descSetLayoutCI.pBindings=descBindings;

  vkCreateDescriptorSetLayout(gVulkanContext.device, &descSetLayoutCI, NULL,
                              &descSetLayout);
  }

  // Creating descriptor pool
  {
  VkDescriptorPoolSize descPoolCapacity[2]={};
  descPoolCapacity[0].type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descPoolCapacity[0].descriptorCount=1;
  descPoolCapacity[1].type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descPoolCapacity[1].descriptorCount=1;
  
  VkDescriptorPoolCreateInfo descPoolCI={};
  descPoolCI.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descPoolCI.maxSets=1;
  descPoolCI.poolSizeCount=sizeof(descPoolCapacity)/
                            sizeof(descPoolCapacity[0]);
  descPoolCI.pPoolSizes=descPoolCapacity;
  vkCreateDescriptorPool(gVulkanContext.device, &descPoolCI, NULL,
                          &gProcessingContext.descPool);
  }

  // Allocating descriptor sets
  VkDescriptorSet descSet;
  {
  VkDescriptorSetAllocateInfo descSetAI={};
  descSetAI.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descSetAI.descriptorPool=gProcessingContext.descPool;
  descSetAI.descriptorSetCount=1;
  descSetAI.pSetLayouts=&descSetLayout;
  vkAllocateDescriptorSets(gVulkanContext.device, &descSetAI, &descSet);
  }

  // Binding our buffers to descriptor set
  {
  VkDescriptorBufferInfo descBufferIs[2]={};
  descBufferIs[0].buffer=gProcessingContext.imgBuffer.buf;
  descBufferIs[0].offset=0;
  descBufferIs[0].range=VK_WHOLE_SIZE;

  descBufferIs[1].buffer=gProcessingContext.configBuffer.buf;
  descBufferIs[1].offset=0;
  descBufferIs[1].range=VK_WHOLE_SIZE;
  
  VkWriteDescriptorSet descSetWrites[2]={};
  descSetWrites[0].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descSetWrites[0].dstSet=descSet;
  descSetWrites[0].dstBinding=0;
  descSetWrites[0].dstArrayElement=0;
  descSetWrites[0].descriptorCount=1;
  descSetWrites[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descSetWrites[0].pBufferInfo=&descBufferIs[0];

  descSetWrites[1].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descSetWrites[1].dstSet=descSet;
  descSetWrites[1].dstBinding=1;
  descSetWrites[1].dstArrayElement=0;
  descSetWrites[1].descriptorCount=1;
  descSetWrites[1].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descSetWrites[1].pBufferInfo=&descBufferIs[1];

  vkUpdateDescriptorSets(gVulkanContext.device,
                        sizeof(descSetWrites)/sizeof(descSetWrites[0]),
                        descSetWrites, 0, NULL);
  }

  // Creaing compute pipeline layout
  VkPipelineLayout filtersPipelineLayout;
  {
  VkPipelineLayoutCreateInfo pipelineLayoutCI={};
  pipelineLayoutCI.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount=1;
  pipelineLayoutCI.pSetLayouts=&descSetLayout;
  pipelineLayoutCI.pushConstantRangeCount=0;
  vkCreatePipelineLayout(gVulkanContext.device, &pipelineLayoutCI, NULL,
                          &filtersPipelineLayout);
  }

  // Creating compute pipeline
  {
  VkShaderModule filtersShader=
      CreateShaderModuleFromIR((uint32_t*)filtersShaderIR, filtersShaderIRSize);
  /*VkShaderModule filtersShader=
      CreateShaderModuleFromFile("shaders/filters.spv");*/
  
  const uint32_t imgSizeSpecConstVal=imgSize;
  
  VkSpecializationMapEntry imgSizeSpecConstMap={};
  imgSizeSpecConstMap.constantID=0;
  imgSizeSpecConstMap.offset=0;
  imgSizeSpecConstMap.size=sizeof(uint32_t);

  VkSpecializationInfo imgSizeSpecConstInfo={};
  imgSizeSpecConstInfo.mapEntryCount=1;
  imgSizeSpecConstInfo.pMapEntries=&imgSizeSpecConstMap;
  imgSizeSpecConstInfo.dataSize=sizeof(imgSizeSpecConstVal);
  imgSizeSpecConstInfo.pData=&imgSizeSpecConstVal;

  VkPipelineShaderStageCreateInfo shaderStageCI={};
  shaderStageCI.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageCI.stage=VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCI.module=filtersShader;
  shaderStageCI.pName="main";
  shaderStageCI.pSpecializationInfo=&imgSizeSpecConstInfo;

  VkComputePipelineCreateInfo pipelineCI={};
  pipelineCI.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.stage=shaderStageCI;
  pipelineCI.layout=filtersPipelineLayout;
  
  vkCreateComputePipelines(gVulkanContext.device, VK_NULL_HANDLE, 1,
                            &pipelineCI, NULL,
                            &gProcessingContext.filtersPipeline);
  vkDestroyShaderModule(gVulkanContext.device, filtersShader, NULL);
  }

  /// NEW:
  // Creating query pool for timestamps
  {
  VkQueryPoolCreateInfo queryPoolCI={};
  queryPoolCI.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  queryPoolCI.queryType=VK_QUERY_TYPE_TIMESTAMP;
  queryPoolCI.queryCount=2;
  vkCreateQueryPool(gVulkanContext.device, &queryPoolCI, NULL,
                    &gProcessingContext.queryPool);
  }
  /// END OF NEW

  // Creating command buffers pool and allocating single command buffer
  // from it
  {
  VkCommandPoolCreateInfo cmdPoolCI={};
  cmdPoolCI.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmdPoolCI.queueFamilyIndex=gVulkanContext.queueFamilyIdx;
  vkCreateCommandPool(gVulkanContext.device, &cmdPoolCI, NULL,
                      &gProcessingContext.cmdPool);

  VkCommandBufferAllocateInfo cmdBufferAI={};
  cmdBufferAI.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdBufferAI.commandPool=gProcessingContext.cmdPool;
  cmdBufferAI.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdBufferAI.commandBufferCount=1;
  vkAllocateCommandBuffers(gVulkanContext.device, &cmdBufferAI,
                            &gProcessingContext.filtersCmdBuf);
  }

  // Recording command to command buffer
  {
  VkCommandBufferBeginInfo cmdBufferBI={};
  cmdBufferBI.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(gProcessingContext.filtersCmdBuf, &cmdBufferBI);

  // Resetting query pool
  vkCmdResetQueryPool(gProcessingContext.filtersCmdBuf,
                      gProcessingContext.queryPool, 0, 2);

  vkCmdBindPipeline(gProcessingContext.filtersCmdBuf,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    gProcessingContext.filtersPipeline);
  vkCmdBindDescriptorSets(gProcessingContext.filtersCmdBuf,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          filtersPipelineLayout, /*firstSet=*/0,
                          /*descriptorSetCount=*/1, &descSet,
                          /*dynamicOffsetsCount=*/0,
                          /*pDynamicOffsets*/NULL);
  
  vkCmdWriteTimestamp(gProcessingContext.filtersCmdBuf,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      gProcessingContext.queryPool, 0);
  vkCmdDispatch(gProcessingContext.filtersCmdBuf,
                (imgSize+WORKGROUP_SIZE-1)/WORKGROUP_SIZE, 1, 1);
  vkCmdWriteTimestamp(gProcessingContext.filtersCmdBuf,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      gProcessingContext.queryPool, 1);

  vkEndCommandBuffer(gProcessingContext.filtersCmdBuf);
  }

  vkDestroyPipelineLayout(gVulkanContext.device, filtersPipelineLayout,
                          NULL);
  vkDestroyDescriptorSetLayout(gVulkanContext.device, descSetLayout, NULL);
  
  {
  VkFenceCreateInfo fenceCI={};
  fenceCI.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(gVulkanContext.device, &fenceCI, NULL,
                &gProcessingContext.fence);
  }

  gProcessingContext.submitInfo={};
  gProcessingContext.submitInfo.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
  gProcessingContext.submitInfo.commandBufferCount=1;
  gProcessingContext.submitInfo.pCommandBuffers=
      &gProcessingContext.filtersCmdBuf;
}

void FinalizeProcessingContext() {
  vkDestroyFence(gVulkanContext.device, gProcessingContext.fence, NULL);
  vkDestroyCommandPool(gVulkanContext.device, gProcessingContext.cmdPool, NULL);
  vkDestroyQueryPool(gVulkanContext.device, gProcessingContext.queryPool, NULL);
  vkDestroyPipeline(gVulkanContext.device, gProcessingContext.filtersPipeline,
                    NULL);
  vkDestroyDescriptorPool(gVulkanContext.device, gProcessingContext.descPool,
                          NULL);
  vmaDestroyBuffer(gVulkanContext.allocator,
                   gProcessingContext.configBuffer.buf,
                   gProcessingContext.configBuffer.alloc);
  vmaDestroyBuffer(gVulkanContext.allocator, gProcessingContext.imgBuffer.buf,
                   gProcessingContext.imgBuffer.alloc);
}

vector<Color> ProcessImageGpu(const vector<Color> &img, const Config config,
                              float *computeOnlyElapsedTime,
                              chrono::milliseconds *allElapsedTime) {
  vector<Color> result(img.size());
  struct {
    double scale, brightnessShift, contrastFactor, gain;
    float blackLevel;
  } configCpuBuffer;
  configCpuBuffer.scale=config.saturation.scale;
  configCpuBuffer.brightnessShift=config.brightnessContrast.brightness/127.0;
  configCpuBuffer.contrastFactor=
      tan((config.brightnessContrast.contrast/127.0+1.0)*(M_PI/4.0));
  configCpuBuffer.gain=1.0f/max(exp(-(double)config.exposure.exposure)-
                                config.exposure.blackLevel, 0.000001);
  configCpuBuffer.blackLevel=config.exposure.blackLevel;
  
  
  auto startTransfer=chrono::high_resolution_clock::now();
  vmaCopyMemoryToAllocation(gVulkanContext.allocator, img.data(),
                            gProcessingContext.imgBuffer.alloc, 0,
                            img.size()*sizeof(Color));
  vmaCopyMemoryToAllocation(gVulkanContext.allocator, &configCpuBuffer,
                            gProcessingContext.configBuffer.alloc, 0,
                            sizeof(configCpuBuffer));
  
  // Submitting command buffer to execution
  vkQueueSubmit(gVulkanContext.queue, 1, &gProcessingContext.submitInfo,
                gProcessingContext.fence);
  vkWaitForFences(gVulkanContext.device, 1, &gProcessingContext.fence, VK_TRUE,
                  ~(uint64_t)0);

  vmaCopyAllocationToMemory(gVulkanContext.allocator,
                            gProcessingContext.imgBuffer.alloc, 0,
                            result.data(), img.size()*sizeof(Color));
  auto endTransfer=chrono::high_resolution_clock::now();

  vkResetFences(gVulkanContext.device, 1, &gProcessingContext.fence);
  
  uint64_t timestamps[2];
  vkGetQueryPoolResults(gVulkanContext.device, gProcessingContext.queryPool, 0,
                        2, sizeof(timestamps), timestamps, sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT);
  timestamps[0]&=gVulkanContext.timestampMask;
  timestamps[1]&=gVulkanContext.timestampMask;
  
  if (computeOnlyElapsedTime) {
    static const float timestampPeriod=[]() -> float {
      VkPhysicalDeviceProperties deviceProps;
      vkGetPhysicalDeviceProperties(gVulkanContext.physDevice, &deviceProps);
      return deviceProps.limits.timestampPeriod/1000000.0f;
    }();
    *computeOnlyElapsedTime=(timestamps[1]-timestamps[0])*timestampPeriod;
  }
  if (allElapsedTime)
    *allElapsedTime=
        chrono::duration_cast<chrono::milliseconds>(endTransfer-startTransfer);
  return result;
}

int main(int argc, const char **argv) {
  try {
    if (argc!=3)
      throw std::runtime_error("Wrong arguments or files not found");
    
    InitializeVulkan();
    
    size_t width{}, height{};
    const vector<Color> img=LoadImage(argv[1], width, height);
    const Config config=LoadConfig(argv[2]);

    // Cpu part:
    vector<Color> CpuResultImg=ProcessImageCpu(img, config, nullptr);
    SaveImage(CpuResultImg, width, height, "gold.png");
    // Gpu part:
    InitializeProcessingContext(img.size());
    vector<Color> GpuResultImg=ProcessImageGpu(img, config, nullptr, nullptr);
    SaveImage(GpuResultImg, width, height, "gpgpu.png");

    // Comparison of images part:
    size_t diffPixelsCount=0;
    uint8_t maxChanDiff=0;
    for (size_t pixI=0; pixI<CpuResultImg.size(); ++pixI) {
      enum ChanNames {R, G, B, A};
      const struct {
        uint8_t channels[4];
      } CpuPixel={TO_U8(CpuResultImg[pixI].r),
                  TO_U8(CpuResultImg[pixI].g),
                  TO_U8(CpuResultImg[pixI].b),
                  TO_U8(CpuResultImg[pixI].a)},
        GpuPixel={TO_U8(GpuResultImg[pixI].r),
                  TO_U8(GpuResultImg[pixI].g),
                  TO_U8(GpuResultImg[pixI].b),
                  TO_U8(GpuResultImg[pixI].a)};
      
      diffPixelsCount+=CpuPixel.channels[R]!=GpuPixel.channels[R] ||
                       CpuPixel.channels[G]!=GpuPixel.channels[G] ||
                       CpuPixel.channels[B]!=GpuPixel.channels[B] ||
                       CpuPixel.channels[A]!=GpuPixel.channels[A];

      for (unsigned chanI=0; chanI<4; ++chanI)
        maxChanDiff=max((uint8_t)abs((int)CpuPixel.channels[chanI]-
                                 GpuPixel.channels[chanI]),
                        maxChanDiff);
    }
    printf("GPU:\n"
           "  number of differing pixels:\t\t%zu (== %f of all pixels)\n"
           "  max difference between channels:\t%u\n",
           diffPixelsCount, (double)diffPixelsCount/img.size(), maxChanDiff);
    
    // Cpu performance measurement:
    for (unsigned warmI=0; warmI<N_WARMUP; ++warmI)
      ProcessImageCpu(img, config, nullptr);
    vector<chrono::milliseconds> CpuTimings(N_ITERS);
    for (unsigned i=0; i<N_ITERS; ++i)
      ProcessImageCpu(img, config, &CpuTimings[i]);
    
    // Gpu performance measurement:
    for (unsigned warmI=0; warmI<N_WARMUP; ++warmI)
      ProcessImageGpu(img, config, nullptr, nullptr);
    vector<float> GpuComputeOnlyTimings(N_ITERS);
    vector<chrono::milliseconds> GpuAllTimings(N_ITERS);
    for (unsigned i=0; i<N_ITERS; ++i)
      ProcessImageGpu(img, config, &GpuComputeOnlyTimings[i],
                                   &GpuAllTimings[i]);
    
    TimeVals CpuTVals, GpuComputeOnlyTVals, GpuAllTVals;
    CalcTimeVals(CpuTVals, CpuTimings, N_SKIP);
    //
    CalcTimeVals(GpuComputeOnlyTVals, GpuComputeOnlyTimings, N_SKIP);
    CalcTimeVals(GpuAllTVals, GpuAllTimings, N_SKIP);

    printf(
      "Performance measurements:\n"
      "CPU:\t\t\t\t\t%10.3f +/- %f MPix/s\n\n"
      "GPU (only computations):\t\t%10.3f +/- %f MPix/s\n"
      "GPU (computations and transfers):\t%10.3f +/- %f MPix/s\n\n",
      img.size()*CpuTVals.mean/CpuTVals.perfDenom/1000,
          img.size()*CpuTVals.Mse/CpuTVals.perfDenom/1000,
      //
      img.size()*GpuComputeOnlyTVals.mean/GpuComputeOnlyTVals.perfDenom/1000,
          img.size()*GpuComputeOnlyTVals.Mse/GpuComputeOnlyTVals.perfDenom/1000,
      img.size()*GpuAllTVals.mean/GpuAllTVals.perfDenom/1000,
          img.size()*GpuAllTVals.Mse/GpuAllTVals.perfDenom/1000);
    
    FinalizeProcessingContext();
    FinalizeVulkan();
    return 0;
  }
  catch (std::exception &ex) {
    fprintf(stderr, "%s\n", ex.what());
    return 42;
  }
}
