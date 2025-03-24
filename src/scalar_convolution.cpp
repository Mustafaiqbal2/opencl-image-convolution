#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <string>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <numeric>

// Include STB image library
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

namespace fs = std::filesystem;

// Function declarations
bool loadImage(const std::string& filename, std::vector<float>& image, int& width, int& height);
void generateRandomImage(std::vector<float>& image, int width, int height);
void saveImagePNG(const std::vector<float>& image, int width, int height, const std::string& filename);
void scalarConvolution(const std::vector<float>& input, std::vector<float>& output, 
                      const std::vector<float>& kernel, int width, int height, int kernelSize);

int main(int argc, char* argv[]) {
    // Default parameters
    int defaultWidth = 1024;
    int defaultHeight = 1024;
    std::string imagesDir = "images";
    bool useRandomData = true;
    bool benchmarkMode = false;
    bool verboseOutput = false;
    
    // Process command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dir" && i + 1 < argc) {
            imagesDir = argv[++i];
            useRandomData = false;
        } else if (arg == "--random") {
            useRandomData = true;
        } else if (arg == "--width" && i + 1 < argc) {
            defaultWidth = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            defaultHeight = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmarkMode = true;
        } else if (arg == "--verbose") {
            verboseOutput = true;
        }
    }
    
    // Define vertical edge detection kernel
    const int kernelSize = 3;
    std::vector<float> kernel = {
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f
    };
    
    // Create output directories
    fs::create_directories("output/imgs");
    fs::create_directories("output/benchmark");
    
    // Vector to store timing results
    std::vector<long> scalarTimes;
    std::vector<std::string> imageNames;
    
    // Collect image files from directory
    std::vector<std::string> imageFiles;
    
    if (!useRandomData) {
        try {
            for (const auto& entry : fs::directory_iterator(imagesDir)) {
                if (entry.is_regular_file()) {
                    std::string extension = entry.path().extension().string();
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                    
                    if (extension == ".jpg" || extension == ".jpeg" || 
                        extension == ".png" || extension == ".bmp" || 
                        extension == ".pgm") {
                        imageFiles.push_back(entry.path().string());
                    }
                }
            }
            
            if (imageFiles.empty()) {
                std::cout << "No image files found in directory: " << imagesDir << std::endl;
                std::cout << "Using random image data instead." << std::endl;
                useRandomData = true;
            } else {
                std::cout << "Found " << imageFiles.size() << " image files in directory: " << imagesDir << std::endl;
                if (benchmarkMode && imageFiles.size() > 5) {
                    // Limit to first 5 images in benchmark mode to save time
                    std::cout << "Benchmark mode: Processing only the first 5 images" << std::endl;
                    imageFiles.resize(5);
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error accessing directory: " << e.what() << std::endl;
            std::cout << "Using random image data instead." << std::endl;
            useRandomData = true;
        }
    }
    
    // Setup timing variables
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // Process random data or real images
    if (useRandomData) {
        // Process random data
        std::vector<float> inputImage(defaultWidth * defaultHeight);
        std::vector<float> outputImage(defaultWidth * defaultHeight);
        
        std::cout << "Processing random image data: " << defaultWidth << "x" << defaultHeight << std::endl;
        
        // Generate random image
        generateRandomImage(inputImage, defaultWidth, defaultHeight);
        
        // Process with scalar implementation
        auto start = std::chrono::high_resolution_clock::now();
        scalarConvolution(inputImage, outputImage, kernel, defaultWidth, defaultHeight, kernelSize);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        scalarTimes.push_back(duration);
        imageNames.push_back("random");
        
        if (verboseOutput) {
            std::cout << "Scalar convolution execution time: " << duration << " ms" << std::endl;
        }
        
        // Save the output image
        saveImagePNG(outputImage, defaultWidth, defaultHeight, "output/imgs/output_random_scalar.png");
    } else {
        // Process real images
        int processedCount = 0;
        int totalImages = imageFiles.size();
        
        std::cout << "Processing " << totalImages << " images..." << std::endl;
        
        for (const auto& imageFile : imageFiles) {
            fs::path imagePath(imageFile);
            std::string baseName = imagePath.stem().string();
            
            if (verboseOutput) {
                std::cout << "\nProcessing image: " << imageFile << std::endl;
            } else {
                // Show progress without too much output
                processedCount++;
                std::cout << "\rProcessing images: " << processedCount << "/" << totalImages 
                          << " (" << (processedCount * 100 / totalImages) << "%)" << std::flush;
            }
            
            imageNames.push_back(baseName);
            
            // Load the image
            std::vector<float> inputImage;
            int width, height;
            
            if (loadImage(imageFile, inputImage, width, height)) {
                // Allocate memory for output image
                std::vector<float> outputImage(width * height);
                
                if (verboseOutput) {
                    std::cout << "Image size: " << width << "x" << height << std::endl;
                }
                
                // Apply scalar convolution and measure execution time
                auto start = std::chrono::high_resolution_clock::now();
                scalarConvolution(inputImage, outputImage, kernel, width, height, kernelSize);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                scalarTimes.push_back(duration);
                
                if (verboseOutput) {
                    std::cout << "Scalar convolution execution time: " << duration << " ms" << std::endl;
                }
                
                // Save the output image
                saveImagePNG(outputImage, width, height, "output/imgs/" + baseName + "_scalar.png");
            } else {
                std::cerr << "Failed to load image: " << imageFile << std::endl;
            }
        }
        std::cout << std::endl; // End the progress line
    }
    
    // Calculate total execution time
    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();
    
    // Calculate average processing time
    double avgTime = 0.0;
    if (!scalarTimes.empty()) {
        avgTime = std::accumulate(scalarTimes.begin(), scalarTimes.end(), 0.0) / scalarTimes.size();
    }
    
    // Output summary to CSV file
    std::ofstream csvFile("output/benchmark/scalar_results.csv");
    if (csvFile.is_open()) {
        csvFile << "Implementation,Average Time (ms),Total Time (ms),Images Processed\n";
        csvFile << "Scalar," << avgTime << "," << totalTime << "," << scalarTimes.size() << "\n";
        
        csvFile.close();
        std::cout << "Benchmark results saved to output/benchmark/scalar_results.csv" << std::endl;
    }
    
    // Print summary statistics
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Number of images processed: " << scalarTimes.size() << std::endl;
    std::cout << "Total Scalar Execution Time: " << totalTime << " ms" << std::endl;
    std::cout << "Average Scalar Execution Time: " << avgTime << " ms" << std::endl;
    
    return 0;
}

// Function to load an image from a file
bool loadImage(const std::string& filename, std::vector<float>& image, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    
    if (!data) {
        std::cerr << "Error: Could not load image file: " << filename << std::endl;
        return false;
    }
    
    // Resize image vector
    image.resize(width * height);
    
    // Convert to grayscale and normalize to [0,1]
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            
            if (channels == 1) {
                // Already grayscale
                image[index] = static_cast<float>(data[index]) / 255.0f;
            }
            else if (channels >= 3) {
                // Convert RGB to grayscale using luminance formula
                int pixelIndex = (y * width + x) * channels;
                float r = static_cast<float>(data[pixelIndex]) / 255.0f;
                float g = static_cast<float>(data[pixelIndex + 1]) / 255.0f;
                float b = static_cast<float>(data[pixelIndex + 2]) / 255.0f;
                image[index] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
    }
    
    // Free the loaded image data
    stbi_image_free(data);
    return true;
}

// Function to generate a random image
void generateRandomImage(std::vector<float>& image, int width, int height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < width * height; ++i) {
        image[i] = dist(gen);
    }
}

// Function to save image in PNG format
void saveImagePNG(const std::vector<float>& image, int width, int height, const std::string& filename) {
    // Create directory if it doesn't exist
    fs::path filePath(filename);
    fs::create_directories(filePath.parent_path());
    
    // Convert float [0,1] to byte [0,255]
    std::vector<unsigned char> byteImage(width * height);
    for (int i = 0; i < width * height; ++i) {
        // Clamp values to [0,1] range
        float pixel = std::max(0.0f, std::min(1.0f, image[i]));
        byteImage[i] = static_cast<unsigned char>(pixel * 255.0f);
    }
    
    // Save the image as PNG
    stbi_write_png(filename.c_str(), width, height, 1, byteImage.data(), width);
}

// Scalar convolution implementation
void scalarConvolution(const std::vector<float>& input, std::vector<float>& output, 
                      const std::vector<float>& kernel, 
                      int width, int height, int kernelSize) {
    int kernelRadius = kernelSize / 2;
    
    // For each pixel in the output image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            
            // Apply convolution kernel
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    // Calculate input image coordinates with zero-padding
                    int ix = x + kx - kernelRadius;
                    int iy = y + ky - kernelRadius;
                    
                    // Handle boundaries with zero-padding
                    float pixelValue = 0.0f;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        pixelValue = input[iy * width + ix];
                    }
                    
                    // Multiply pixel value by kernel weight
                    sum += pixelValue * kernel[ky * kernelSize + kx];
                }
            }
            
            // Store result in output image
            output[y * width + x] = sum;
        }
    }
}