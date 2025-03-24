CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
LDFLAGS = -lOpenCL -lstdc++fs

# Directories
SRC_DIR = src
INCLUDE_DIR = $(SRC_DIR)/include
KERNEL_DIR = $(SRC_DIR)/kernels
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
OUTPUT_DIR = output
IMAGE_DIR = images

# Create necessary directories
$(shell mkdir -p $(SRC_DIR) $(INCLUDE_DIR) $(KERNEL_DIR) $(BUILD_DIR) $(BIN_DIR) $(OUTPUT_DIR)/imgs $(OUTPUT_DIR)/benchmark $(IMAGE_DIR))

# Source files
SCALAR_SRC = $(SRC_DIR)/scalar_convolution.cpp
OPENCL_SRC = $(SRC_DIR)/opencl_convolution.cpp

# Executable names
SCALAR_BIN = $(BIN_DIR)/scalar_convolution
OPENCL_BIN = $(BIN_DIR)/opencl_convolution

# OpenCL kernel files (assumed to already exist)
KERNEL_FILES = $(KERNEL_DIR)/convolution_kernel.cl $(KERNEL_DIR)/optimized_convolution_kernel.cl

# STB headers (downloaded if needed)
STB_HEADERS = $(INCLUDE_DIR)/stb_image.h $(INCLUDE_DIR)/stb_image_write.h

# Default target
all: $(STB_HEADERS) $(SCALAR_BIN) $(OPENCL_BIN) copy_kernels

# Download STB headers if not present
$(INCLUDE_DIR)/stb_image.h:
	mkdir -p $(INCLUDE_DIR)
	wget -O $@ https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

$(INCLUDE_DIR)/stb_image_write.h:
	mkdir -p $(INCLUDE_DIR)
	wget -O $@ https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

# Build scalar convolution
$(SCALAR_BIN): $(SCALAR_SRC) $(STB_HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Build OpenCL convolution
$(OPENCL_BIN): $(OPENCL_SRC) $(STB_HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Copy kernel files to bin directory for easy access at runtime
copy_kernels:
	@if [ -f $(KERNEL_DIR)/convolution_kernel.cl ]; then \
	    cp $(KERNEL_DIR)/convolution_kernel.cl $(BIN_DIR)/; \
	else \
	    echo "Warning: $(KERNEL_DIR)/convolution_kernel.cl not found"; \
	fi

# Run targets
run_scalar_random: $(SCALAR_BIN)
	cd $(BIN_DIR) && ./scalar_convolution --random

run_opencl_random: $(OPENCL_BIN) copy_kernels
	cd $(BIN_DIR) && ./opencl_convolution --random

# Run with directory of images - Scalar only
run_scalar_benchmark: $(SCALAR_BIN)
	@echo "Running scalar benchmark on images in $(IMAGE_DIR)"
	cd $(BIN_DIR) && ./scalar_convolution --dir ../../$(IMAGE_DIR)

# Run with directory of images - Scalar only (verbose)
run_scalar_benchmark_verbose: $(SCALAR_BIN)
	@echo "Running verbose scalar benchmark on images in $(IMAGE_DIR)"
	cd $(BIN_DIR) && ./scalar_convolution --dir ../../$(IMAGE_DIR) --verbose

# Run with directory of images - OpenCL
run_opencl_benchmark: $(OPENCL_BIN) copy_kernels
	@echo "Running OpenCL benchmark on images in $(IMAGE_DIR)"
	cd $(BIN_DIR) && ./opencl_convolution --dir ../../$(IMAGE_DIR)

# Run with directory of images - OpenCL (verbose)
run_opencl_benchmark_verbose: $(OPENCL_BIN) copy_kernels
	@echo "Running verbose OpenCL benchmark on images in $(IMAGE_DIR)"
	cd $(BIN_DIR) && ./opencl_convolution --dir ../../$(IMAGE_DIR) --verbose

# Run complete benchmark (both scalar and OpenCL)
run_benchmark: run_scalar_benchmark run_opencl_benchmark
	@echo "Complete benchmark finished"

# Clean targets
clean:
	rm -rf $(BUILD_DIR)/* $(OUTPUT_DIR)/imgs/* $(OUTPUT_DIR)/benchmark/*

clean_all: clean
	rm -rf $(OUTPUT_DIR)/*

.PHONY: all clean clean_all run_scalar_random run_opencl_random run_scalar_benchmark run_scalar_benchmark_verbose run_opencl_benchmark run_opencl_benchmark_verbose run_benchmark copy_kernels