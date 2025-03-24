// Basic convolution kernel
__kernel void convolution(
    __global const float* restrict input,
    __global float* restrict output,
    __constant float* restrict convKernel,
    const int width,
    const int height,
    const int kernelSize)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const int radius = kernelSize / 2;
    float sum = 0.0f;
    
    #pragma unroll 3
    for (int ky = 0; ky < kernelSize; ++ky) {
        const int iy = y + ky - radius;
        
        #pragma unroll 3
        for (int kx = 0; kx < kernelSize; ++kx) {
            const int ix = x + kx - radius;
            
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                sum += input[iy * width + ix] * convKernel[ky * kernelSize + kx];
            }
        }
    }
    
    output[y * width + x] = sum;
}

// High-performance optimized kernel using local memory
__kernel void optimized_convolution(
    __global const float* restrict input,
    __global float* restrict output,
    __constant float* restrict convKernel,
    const int width,
    const int height,
    const int kernelSize,
    __local float* localMem)
{
    const int localX = get_local_id(0);
    const int localY = get_local_id(1);
    const int globalX = get_global_id(0);
    const int globalY = get_global_id(1);
    
    const int localWidth = get_local_size(0);
    const int localHeight = get_local_size(1);
    const int groupIdX = get_group_id(0);
    const int groupIdY = get_group_id(1);
    
    const int radius = kernelSize / 2;
    
    // Calculate tile sizes including halo
    const int tileWidth = localWidth + 2 * radius;
    const int tileHeight = localHeight + 2 * radius;
    
    // Calculate starting points
    const int startX = groupIdX * localWidth - radius;
    const int startY = groupIdY * localHeight - radius;
    
    // Load data into local memory more efficiently in coalesced patterns
    // Each thread loads multiple elements to hide memory latency
    for (int j = 0; j < tileHeight; j += localHeight) {
        int y = j + localY;
        if (y < tileHeight) {
            int inputY = startY + y;
            
            for (int i = 0; i < tileWidth; i += localWidth) {
                int x = i + localX;
                if (x < tileWidth) {
                    int inputX = startX + x;
                    
                    // Clamp to image boundaries
                    float value = 0.0f;
                    if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                        value = input[inputY * width + inputX];
                    }
                    
                    localMem[y * tileWidth + x] = value;
                }
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (globalX < width && globalY < height) {
        float sum = 0.0f;
        
        const int localPosX = localX + radius;
        const int localPosY = localY + radius;
        
        #pragma unroll 3
        for (int ky = 0; ky < kernelSize; ++ky) {
            #pragma unroll 3
            for (int kx = 0; kx < kernelSize; ++kx) {
                sum += localMem[(localPosY - radius + ky) * tileWidth + 
                               (localPosX - radius + kx)] * 
                       convKernel[ky * kernelSize + kx];
            }
        }
        
        output[globalY * width + globalX] = sum;
    }
}