# OpenCL Image Convolution

This project demonstrates the use of OpenCL for parallel image processing by implementing convolution operations on images. It processes a set of input images while comparing the performance of different convolution implementation strategies.

## Overview

The program applies a convolution filter to images using the following process:
- Read input images from a specified directory
- Apply convolution operations using OpenCL on GPU
- Compare basic and optimized kernel implementations
- Save processed images and performance metrics

The implementation enables high-performance image processing by:
1. Utilizing GPU parallel processing capabilities
2. Exploiting local memory for improved performance
3. Comparing different optimization techniques

## Dependencies

- C++ compiler with C++17 support
- OpenCL development libraries
- STB image libraries (included)
- CMake (for building)

## Installation

### Ubuntu/Debian
```bash
# Install dependencies
sudo apt update
sudo apt install g++ ocl-icd-opencl-dev opencl-headers

# Clone the repository
git clone https://github.com/yourusername/opencl-image-convolution.git
cd opencl-image-convolution

# Compile with CMake
mkdir build && cd build
cmake ..
make
```

### macOS
```bash
# Install dependencies (if needed, OpenCL is available by default)
brew install cmake

# Compile with CMake
mkdir build && cd build
cmake ..
make
```

### Windows
```bash
# Using CMake GUI or command line
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Usage

Run the compiled executable with an optional image directory:
```bash
# Using default images directory
./opencl_convolution

# Specify custom image directory
./opencl_convolution --dir /path/to/images
```

The program will:
1. Load images from the specified directory
2. Apply convolution operations using OpenCL
3. Save processed images to the output directory
4. Generate performance benchmark results in CSV format

## Implementation Details

### Convolution Kernels

The project implements two types of OpenCL kernels:

1. **Basic Kernel**:
   - Direct implementation of convolution operation
   - Each work item processes one output pixel
   - Global memory access for all operations

2. **Optimized Kernel**:
   - Uses local memory to reduce global memory access
   - Tiles the input to improve data locality
   - Employs coalesced memory access patterns
   - Takes advantage of GPU architecture for faster processing

### OpenCL Optimizations

- **Memory Management**: Strategic use of different memory types (global, local, constant)
- **Work Group Optimization**: Selection of optimal work group sizes for target devices
- **Memory Access Patterns**: Coalesced memory accesses for improved throughput
- **Compiler Optimizations**: Use of OpenCL compiler flags to enhance performance

## Performance Analysis

The program provides comprehensive performance metrics:
- Execution time for basic and optimized kernels
- Speedup calculations
- Comparison of execution times across different image sizes
- CSV output for further analysis

## Code Structure

- **opencl_convolution.cpp**: Main program handling image I/O and OpenCL setup
- **convolution_kernel.cl**: OpenCL kernel implementations
- **stb_image.h/stb_image_write.h**: Image loading/saving libraries
- **makefile**: Build configuration

## Academic Context

This project demonstrates principles of GPU computing including:
- Parallelization of image processing algorithms
- Memory hierarchy optimization
- Work distribution strategies
- Performance analysis methodologies
- Tradeoffs between different implementation approaches

## Acknowledgments

- The Khronos Group for the OpenCL framework
- STB library developers for the image processing utilities
- The open source community for various resources on GPU optimization