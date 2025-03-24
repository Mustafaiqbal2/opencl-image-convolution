#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <numeric>
#include <CL/cl.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

namespace fs = std::filesystem;

std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::ifstream file2("../kernels/" + filename);
        if (!file2.is_open()) return ""; 
        return std::string(std::istreambuf_iterator<char>(file2), std::istreambuf_iterator<char>());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

bool loadImage(const std::string& filename, std::vector<float>& image, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) return false;
    
    image.resize(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            if (channels == 1) {
                image[index] = static_cast<float>(data[index]) / 255.0f;
            } else if (channels >= 3) {
                int pixelIndex = (y * width + x) * channels;
                float r = static_cast<float>(data[pixelIndex]) / 255.0f;
                float g = static_cast<float>(data[pixelIndex + 1]) / 255.0f;
                float b = static_cast<float>(data[pixelIndex + 2]) / 255.0f;
                image[index] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
    }
    stbi_image_free(data);
    return true;
}

void saveImagePNG(const std::vector<float>& image, int width, int height, const std::string& filename) {
    fs::create_directories(fs::path(filename).parent_path());
    std::vector<unsigned char> byteImage(width * height);
    for (int i = 0; i < width * height; ++i) {
        byteImage[i] = static_cast<unsigned char>(255.0f * std::min(1.0f, std::max(0.0f, image[i])));
    }
    stbi_write_png(filename.c_str(), width, height, 1, byteImage.data(), width);
}

int main(int argc, char* argv[]) {
    // Get image directory
    std::string imagesDir = "images";
    if (argc > 2 && std::string(argv[1]) == "--dir") {
        imagesDir = argv[2];
    }
    
    // Create directories
    fs::create_directories("output/imgs");
    fs::create_directories("output/benchmark");
    
    // Convolution kernel
    const int kernelSize = 3;
    std::vector<float> convKernel = {
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f
    };
    
    // Get all image files
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            for (auto& c : ext) c = std::tolower(c);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".pgm") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (imageFiles.empty()) {
        std::cerr << "No image files found in directory: " << imagesDir << std::endl;
        return 1;
    }
    
    std::cout << "Found " << imageFiles.size() << " image files." << std::endl;
    
    // OpenCL setup
    cl_int err;
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    // Find GPU
    cl_device_id device = nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint numDevices;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) != CL_SUCCESS)
            continue;
        
        if (numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
            
            // Prefer NVIDIA or AMD GPUs
            for (cl_device_id dev : devices) {
                char vendor[256];
                clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
                std::string vendorStr = vendor;
                
                // NVIDIA GPUs typically have better OpenCL perf
                if (vendorStr.find("NVIDIA") != std::string::npos) {
                    device = dev;
                    break;
                } else if (vendorStr.find("AMD") != std::string::npos) {
                    device = dev;
                    // Don't break - continue looking for NVIDIA
                } else if (!device) {
                    device = dev; // Use as fallback
                }
            }
            
            if (!device && !devices.empty()) device = devices[0];
            
            // Get device info
            if (device) {
                char deviceName[256];
                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                
                cl_uint computeUnits;
                clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
                
                std::cout << "Using GPU: " << deviceName << " with " << computeUnits << " compute units" << std::endl;
                break;
            }
        }
    }
    
    if (!device) {
        std::cerr << "No GPU device found!" << std::endl;
        return 1;
    }
    
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    
    // Load kernel source
    std::string kernelSource = readKernelSource("convolution_kernel.cl");
    if (kernelSource.empty()) {
        std::cerr << "Failed to load kernel source" << std::endl;
        return 1;
    }
    
    const char* sourceCStr = kernelSource.c_str();
    size_t sourceLength = kernelSource.length();
    
    cl_program program = clCreateProgramWithSource(context, 1, &sourceCStr, &sourceLength, &err);
    
    // Build with optimization flags
    err = clBuildProgram(program, 1, &device, 
        "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -cl-denorms-are-zero", nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        buildLog[logSize] = '\0';
        std::cerr << "Kernel compilation error:\n" << buildLog.data() << std::endl;
        return 1;
    }
    
    // Create kernels
    cl_kernel basicKernel = clCreateKernel(program, "convolution", &err);
    cl_kernel optimizedKernel = clCreateKernel(program, "optimized_convolution", &err);
    bool useOptimized = (err == CL_SUCCESS);
    
    // Create kernel buffer
    cl_mem kernelBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * convKernel.size(), convKernel.data(), &err);
    
    // Get device limits for tuning
    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    
    cl_ulong localMemSize;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr);
    
    // Find optimal work group size
    // For modern GPUs, 16x16 or 32x8 tends to work well
    size_t bestWorkGroupSizes[][2] = {{32, 8}, {16, 16}, {32, 4}, {8, 32}, {8, 8}};
    size_t localWorkSize[2] = {16, 16}; // Default
    
    // Find largest valid work group size
    for (auto& wgSize : bestWorkGroupSizes) {
        if (wgSize[0] * wgSize[1] <= maxWorkGroupSize) {
            localWorkSize[0] = wgSize[0];
            localWorkSize[1] = wgSize[1];
            break;
        }
    }
    
    // Pre-allocate vectors for the entire batch to reduce memory allocations
    std::vector<double> basicTimes, optimizedTimes;
    basicTimes.reserve(imageFiles.size());
    optimizedTimes.reserve(imageFiles.size());
    
    // Process images
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    int processedCount = 0;
    
    // Split work into batches for better memory management
    const size_t batchSize = 250; // Process in batches
    size_t numBatches = (imageFiles.size() + batchSize - 1) / batchSize;
    
    std::cout << "Processing images in " << numBatches << " batches..." << std::endl;
    
    for (size_t batch = 0; batch < numBatches; ++batch) {
        size_t start = batch * batchSize;
        size_t end = std::min(start + batchSize, imageFiles.size());
        
        for (size_t i = start; i < end; ++i) {
            const auto& imageFile = imageFiles[i];
            fs::path imagePath(imageFile);
            std::string baseName = imagePath.stem().string();
            
            // Show progress
            processedCount++;
            std::cout << "\rProcessing: " << processedCount << "/" << imageFiles.size() 
                    << " (" << (processedCount * 100 / imageFiles.size()) << "%)" << std::flush;
            
            // Load image
            std::vector<float> inputImage;
            int width, height;
            
            if (loadImage(imageFile, inputImage, width, height)) {
                // Skip small images
                if (width < 32 || height < 32) continue;
                
                // Allocate output images
                std::vector<float> outputBasic(width * height);
                std::vector<float> outputOptimized(width * height);
                
                // Create OpenCL buffers
                cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                sizeof(float) * inputImage.size(), inputImage.data(), &err);
                
                cl_mem outputBasicBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * outputBasic.size(), nullptr, &err);
                
                // Basic kernel arguments
                clSetKernelArg(basicKernel, 0, sizeof(cl_mem), &inputBuffer);
                clSetKernelArg(basicKernel, 1, sizeof(cl_mem), &outputBasicBuffer);
                clSetKernelArg(basicKernel, 2, sizeof(cl_mem), &kernelBuffer);
                clSetKernelArg(basicKernel, 3, sizeof(int), &width);
                clSetKernelArg(basicKernel, 4, sizeof(int), &height);
                clSetKernelArg(basicKernel, 5, sizeof(int), &kernelSize);
                
                // Global work size must be multiple of local work size
                size_t globalWorkSize[2] = {
                    ((width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0],
                    ((height + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1]
                };
                
                // Execute basic kernel (don't save time if we have optimized)
                cl_event kernelEvent;
                auto startBasic = std::chrono::high_resolution_clock::now();
                
                err = clEnqueueNDRangeKernel(queue, basicKernel, 2, nullptr, globalWorkSize, 
                                        localWorkSize, 0, nullptr, &kernelEvent);
                
                clWaitForEvents(1, &kernelEvent);
                clReleaseEvent(kernelEvent);
                
                auto endBasic = std::chrono::high_resolution_clock::now();
                double basicTime = std::chrono::duration_cast<std::chrono::microseconds>
                                  (endBasic - startBasic).count() / 1000.0;
                basicTimes.push_back(basicTime);
                
                // Read back basic result
                clEnqueueReadBuffer(queue, outputBasicBuffer, CL_TRUE, 0,
                                sizeof(float) * outputBasic.size(), outputBasic.data(), 0, nullptr, nullptr);
                
                // Save only every 100th image to reduce I/O
                if (processedCount % 100 == 0) {
                    saveImagePNG(outputBasic, width, height, "output/imgs/" + baseName + "_opencl_basic.png");
                }
                
                // Optimized kernel
                if (useOptimized) {
                    cl_mem outputOptimizedBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                        sizeof(float) * outputOptimized.size(), nullptr, &err);
                    
                    // Calculate local memory size
                    const int radius = kernelSize / 2;
                    size_t tileWidth = localWorkSize[0] + 2 * radius;
                    size_t tileHeight = localWorkSize[1] + 2 * radius;
                    size_t localMemSize = tileWidth * tileHeight * sizeof(float);
                    
                    clSetKernelArg(optimizedKernel, 0, sizeof(cl_mem), &inputBuffer);
                    clSetKernelArg(optimizedKernel, 1, sizeof(cl_mem), &outputOptimizedBuffer);
                    clSetKernelArg(optimizedKernel, 2, sizeof(cl_mem), &kernelBuffer);
                    clSetKernelArg(optimizedKernel, 3, sizeof(int), &width);
                    clSetKernelArg(optimizedKernel, 4, sizeof(int), &height);
                    clSetKernelArg(optimizedKernel, 5, sizeof(int), &kernelSize);
                    clSetKernelArg(optimizedKernel, 6, localMemSize, nullptr);
                    
                    // Execute optimized kernel
                    auto startOpt = std::chrono::high_resolution_clock::now();
                    
                    cl_event optEvent;
                    err = clEnqueueNDRangeKernel(queue, optimizedKernel, 2, nullptr, globalWorkSize, 
                                            localWorkSize, 0, nullptr, &optEvent);
                    
                    clWaitForEvents(1, &optEvent);
                    clReleaseEvent(optEvent);
                    
                    auto endOpt = std::chrono::high_resolution_clock::now();
                    double optTime = std::chrono::duration_cast<std::chrono::microseconds>
                                    (endOpt - startOpt).count() / 1000.0;
                    optimizedTimes.push_back(optTime);
                    
                    // Read back optimized result
                    clEnqueueReadBuffer(queue, outputOptimizedBuffer, CL_TRUE, 0,
                                    sizeof(float) * outputOptimized.size(), outputOptimized.data(),
                                    0, nullptr, nullptr);
                    
                    // Save only every 100th image to reduce I/O
                    if (processedCount % 100 == 0) {
                        saveImagePNG(outputOptimized, width, height, 
                                "output/imgs/" + baseName + "_opencl_optimized.png");
                    }
                    
                    clReleaseMemObject(outputOptimizedBuffer);
                }
                
                clReleaseMemObject(outputBasicBuffer);
                clReleaseMemObject(inputBuffer);
            }
        }
        
        // Force garbage collection
        clFinish(queue);
    }
    std::cout << std::endl;

    // Calculate stats
    auto totalEndTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEndTime - totalStartTime).count();
    
    double avgTime = 0.0;
    if (!basicTimes.empty()) {
        // Calculate the correct average - divide total time by number of images
        avgTime = (double)totalTime / processedCount;
    }
    
    // Write results
    std::ofstream csvFile("output/benchmark/opencl_results.csv");
    if (csvFile.is_open()) {
        csvFile << "Images Processed,Total Time (ms),Average Time (ms)\n";
        csvFile << processedCount << "," << totalTime << "," << avgTime << "\n";
        csvFile.close();
    }
    
    // Print summary in the desired format
    std::cout << "\nBenchmark results saved to output/benchmark/opencl_results.csv" << std::endl;
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Number of images processed: " << processedCount << std::endl;
    std::cout << "Total OpenCL Execution Time: " << totalTime << " ms" << std::endl;
    std::cout << "Average OpenCL Execution Time: " << avgTime << " ms" << std::endl;
    double avgKernelTime = 0.0;
    if (!basicTimes.empty()) {
        avgKernelTime = std::accumulate(basicTimes.begin(), basicTimes.end(), 0.0) / basicTimes.size();
    }

    std::cout << "Average OpenCL Kernel Execution Time: " << avgKernelTime << " ms" << std::endl;
       
    // Clean up
    clReleaseMemObject(kernelBuffer);
    clReleaseKernel(basicKernel);
    if (useOptimized) clReleaseKernel(optimizedKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}