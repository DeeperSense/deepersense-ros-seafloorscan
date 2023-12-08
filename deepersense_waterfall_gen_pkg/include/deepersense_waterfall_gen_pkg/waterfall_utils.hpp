#pragma once 

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <vector>

#include "math_constants.h"

#define BLOCK_SIZE  256
#define COARSENING  4

namespace utils 
{

/**
 * @brief Last warp unrolling for minimum calculation of 2D vectors 
 * 
 * @tparam blockSize Number of threads in each block
 * @tparam T Variable type 
 * @param data Matrix of 2d vectors 
 * @param tid Thread index 
 * @return __device__ 
 */
template<unsigned int blockSize, typename T>
__device__ void warpMin(Eigen::Matrix<T, 2, 1>* data, 
                        unsigned int tid)
{
    if (blockSize >= 64) 
    {
        data[tid](0) = min(data[tid](0), data[tid + 32](0));
        data[tid](1) = min(data[tid](1), data[tid + 32](1));
        __syncthreads();
    }
    if (blockSize >= 32) 
    {
        data[tid](0) = min(data[tid](0), data[tid + 16](0));
        data[tid](1) = min(data[tid](1), data[tid + 16](1));
        __syncthreads();
    }
    if (blockSize >= 16)
    {
        data[tid](0) = min(data[tid](0), data[tid + 8](0));
        data[tid](1) = min(data[tid](1), data[tid + 8](1));
        __syncthreads();
    }
    if (blockSize >=  8) 
    {
        data[tid](0) = min(data[tid](0), data[tid + 4](0));
        data[tid](1) = min(data[tid](1), data[tid + 4](1));
        __syncthreads();
    }
    if (blockSize >=  4)
    {
        data[tid](0) = min(data[tid](0), data[tid + 2](0));
        data[tid](1) = min(data[tid](1), data[tid + 2](1));
        __syncthreads();
    }
    if (blockSize >=  2) 
    {
        data[tid](0) = min(data[tid](0), data[tid + 1](0));
        data[tid](1) = min(data[tid](1), data[tid + 1](1));
        __syncthreads();
    }
} 

/**
 * @brief Last warp unrolling for maximum calculation of 2D vectors 
 * 
 * @tparam blockSize Number of threads in each block
 * @tparam T Variable type 
 * @param data Matrix of 2d vectors 
 * @param tid Thread index 
 * @return __device__ 
 */
template<unsigned int blockSize, typename T>
__device__ void warpMax(Eigen::Matrix<T, 2, 1>* data, 
                        unsigned int tid)
{
    if (blockSize >= 64) 
    {
        data[tid](0) = max(data[tid](0), data[tid + 32](0));
        data[tid](1) = max(data[tid](1), data[tid + 32](1));
        __syncthreads();
    }

    if (blockSize >= 32) 
    {
        data[tid](0) = max(data[tid](0), data[tid + 16](0));
        data[tid](1) = max(data[tid](1), data[tid + 16](1));
        __syncthreads();
    }
    if (blockSize >= 16)
    {
        data[tid](0) = max(data[tid](0), data[tid + 8](0));
        data[tid](1) = max(data[tid](1), data[tid + 8](1));
        __syncthreads();
    }
    if (blockSize >=  8) 
    {
        data[tid](0) = max(data[tid](0), data[tid + 4](0));
        data[tid](1) = max(data[tid](1), data[tid + 4](1));
        __syncthreads();
    }
    if (blockSize >=  4)
    {
        data[tid](0) = max(data[tid](0), data[tid + 2](0));
        data[tid](1) = max(data[tid](1), data[tid + 2](1));
        __syncthreads();
    }
    if (blockSize >=  2) 
    {
        data[tid](0) = max(data[tid](0), data[tid + 1](0));
        data[tid](1) = max(data[tid](1), data[tid + 1](1));
        __syncthreads();
    }
} 

/**
 * @brief Parallel reduction for minimum calculation of 2D vectors 
 * 
 * @tparam blockSize Number of threads in each block
 * @tparam T Variable type
 * @param input Matrix of 2d vectors 
 * @param output Matrix of 2d vectors 
 * @param n Number of vectors
 * @return __global__ 
 */
template<unsigned int blockSize, typename T>
__global__ void minReduce(Eigen::Matrix<T, 2, 1>* input, 
                          Eigen::Matrix<T, 2, 1>* output, 
                          int n)
{
    __shared__ Eigen::Matrix<T, 2, 1> data[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;

    if (tid < blockSize)
    {
        size_t i = blockIdx.x * (blockSize) + tid;
        size_t gridSize = blockSize * gridDim.x;
        
        Eigen::Matrix<T, 2, 1> initial;
        initial << FLT_MAX, FLT_MAX;

        data[tid] = initial;

        while (i < n) 
        { 
            data[tid](0) = min(data[tid](0), input[i](0));
            data[tid](1) = min(data[tid](1), input[i](1)); 
            i += gridSize; 
        }
        __syncthreads();

        if (blockSize >= 512) 
        {
            if (tid < 256)
            {
                data[tid](0) = min(data[tid](0), data[tid + 256](0));
                data[tid](1) = min(data[tid](1), data[tid + 256](1));
            }
            __syncthreads();
        }
        if (blockSize >= 256)
        {
            if (tid < 128)
            {
                data[tid](0) = min(data[tid](0), data[tid + 128](0));
                data[tid](1) = min(data[tid](1), data[tid + 128](1));
            }
            __syncthreads();
        }
        if (blockSize >=  128) 
        {
            if (tid < 64)
            {
                data[tid](0) = min(data[tid](0), data[tid + 64](0));
                data[tid](1) = min(data[tid](1), data[tid + 64](1));
            }
            __syncthreads();
        }

        if (tid < 32) warpMin<blockSize>(data, tid);
        if (tid == 0) { output[blockIdx.x] = data[0]; }  
    }
}

/**
 * @brief Parallel reduction for maximum calculation of 2D vectors 
 * 
 * @tparam blockSize Number of threads in each block
 * @tparam T Variable type
 * @param input Matrix of 2d vectors 
 * @param output Matrix of 2d vectors 
 * @param n Number of vectors
 * @return __global__ 
 */
template<unsigned int blockSize, typename T>
__global__ void maxReduce(Eigen::Matrix<T, 2, 1>* input, 
                          Eigen::Matrix<T, 2, 1>* output, 
                          int n)
{
    __shared__ Eigen::Matrix<T, 2, 1> data[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;

    if (tid < blockSize)
    {
        size_t i = blockIdx.x * (blockSize) + tid;
        size_t gridSize = blockSize * gridDim.x;

        Eigen::Matrix<T, 2, 1> initial;
        initial << FLT_MIN, FLT_MIN;

        data[tid] = initial;

        while (i < n) 
        { 
            data[tid](0) = max(data[tid](0), input[i](0));
            data[tid](1) = max(data[tid](1), input[i](1)); 
            i += gridSize; 
        }
        __syncthreads();

        if (blockSize >= 512) 
        {
            if (tid < 256)
            {
                data[tid](0) = max(data[tid](0), data[tid + 256](0));
                data[tid](1) = max(data[tid](1), data[tid + 256](1));
            }
            __syncthreads();
        }
        if (blockSize >= 256)
        {
            if (tid < 128)
            {
                data[tid](0) = max(data[tid](0), data[tid + 128](0));
                data[tid](1) = max(data[tid](1), data[tid + 128](1));
            }
            __syncthreads();
        }
        if (blockSize >=  128) 
        {
            if (tid < 64)
            {
                data[tid](0) = max(data[tid](0), data[tid + 64](0));
                data[tid](1) = max(data[tid](1), data[tid + 64](1));
            }
            __syncthreads();
        }

        if (tid < 32) warpMax<blockSize>(data, tid);
        if (tid == 0) { output[blockIdx.x] = data[0]; }
    }
}

/**
 * @brief Calculation of minimum and maximum in vector of 2D points on GPU
 * 
 * @tparam blockSize Number of threads in each block
 * @tparam T Variable type
 * @param input Matrix of 2D points
 * @param N Number of points
 * @param min Output minimum point
 * @param max Output maximum point
 */
template<unsigned int blockSize, typename T>
void findMinMax(Eigen::Matrix<T, 2, 1>* input, 
                int N,
                Eigen::Matrix<T, 2, 1>& min, 
                Eigen::Matrix<T, 2, 1>& max)
{
    typedef Eigen::Matrix<T, 2, 1> Vector2;

    size_t blocksPerGrid = ceil((1.0 * N) / BLOCK_SIZE);

    Vector2* tmpMin;
    cudaMalloc(&tmpMin, sizeof(Vector2) * blocksPerGrid); 

    Vector2* fromMin;
    cudaMalloc((void **)&fromMin, N * sizeof(Vector2));
    cudaMemcpy(fromMin, &input[0], N * sizeof(Vector2), cudaMemcpyHostToDevice);

    Vector2* tmpMax;
    cudaMalloc(&tmpMax, sizeof(Vector2) * blocksPerGrid); 

    Vector2* fromMax;
    cudaMalloc((void **)&fromMax, N * sizeof(Vector2));
    cudaMemcpy(fromMax, &input[0], N * sizeof(Vector2), cudaMemcpyHostToDevice);

    do 
    {
        blocksPerGrid  = ceil((1.0 * N) / BLOCK_SIZE);

        minReduce<BLOCK_SIZE, T><<<blocksPerGrid, BLOCK_SIZE>>>(fromMin, tmpMin, N);
        maxReduce<BLOCK_SIZE, T><<<blocksPerGrid, BLOCK_SIZE>>>(fromMax, tmpMax, N);
        
        N = blocksPerGrid;

        fromMax = tmpMax;
        fromMin = tmpMin;
    } 
    while (N > BLOCK_SIZE);

    if (N > 1)
    {
        minReduce<BLOCK_SIZE, T><<<1, BLOCK_SIZE>>>(tmpMin, tmpMin, N);
        maxReduce<BLOCK_SIZE, T><<<1, BLOCK_SIZE>>>(tmpMax, tmpMax, N);
    }

    cudaMemcpy(min.data(), tmpMin, sizeof(Eigen::Matrix<T, 2, 1>), cudaMemcpyDeviceToHost); 
    cudaMemcpy(max.data(), tmpMax, sizeof(Eigen::Matrix<T, 2, 1>), cudaMemcpyDeviceToHost); 

    cudaFree(tmpMin);
    cudaFree(fromMin);
    cudaFree(tmpMax);
    cudaFree(fromMax);
}

/**
 * @brief Calculates the swath positions 
 * 
 * @param yaw Yaw in radians
 * @param altitude Altitude in metres
 * @param east UTM x coordinate 
 * @param north UTM y coordinate 
 * @param slantRes Slant resolution 
 * @param numSamples Number of samples
 * @param swathBins Bin indices in swath
 * @param swathBinsFromCentre Bin indices in swath from centre bin
 * @param line 
 */
void createSwath(float yaw, 
                 float altitude, 
                 double east, 
                 double north, 
                 float slantRes, 
                 int numSamples, 
                 const std::vector<int>& swathBins, 
                 const std::vector<float>& swathBinsFromCentre, 
                 std::vector<Eigen::Vector2d>& line);

/**
 * @brief Corrects the distorted waterfall 
 * 
 * @param numLines Number of pings
 * @param numPoints Number of samples
 * @param resolution Map resolution
 * @param slantRes Slant resolution 
 * @param maxNeighbours Maximum number of neighbours for interpolation
 * @param swathBins Bin indices in swath
 * @param swathBinsFromCentre Bin indices in swath from centre bin
 * @param swathData (UTM x coordinate, y coordinate, altitude, yaw) per swath
 * @param originalLines Original swath positions
 * @param originalIntensities Original waterfall intensities
 * @param waterColumnMask Water-column mask 
 * @param correctedLines Corrected swath positions 
 * @param correctedIntensities Corrected waterfall intensities 
 */
void interpolate(int numLines, 
                 int numPoints,  
                 float resolution,
                 float slantRes, 
                 int maxNeighbours,
                 const std::vector<int>& swathBins, 
                 const std::vector<float>& swathBinsFromCentre, 
                 const std::vector<Eigen::Vector4d>& swathData, 
                 const std::vector<Eigen::Vector2d>& originalLines, 
                 const std::vector<int>& originalIntensities,
                 const std::vector<int>& waterColumnMask,
                 std::vector<Eigen::Vector2d>& correctedLines, 
                 std::vector<int>& correctedIntensities);


/* CUDA KERNELS */

/**
 * @brief Corrects the swath positions (CUDA kernel)
 * 
 * @param headingFirst First yaw in radians
 * @param headingDelta Yaw delta in radians 
 * @param data (UTM x coordinate, y coordinate, altitude, yaw) per swath
 * @param bins Bin indices in swath
 * @param binsFromCentre Bin indices in swath from centre bin
 * @param slantRes Slant resolution 
 * @param numPoints Number of samples
 * @param numLines Number of pings
 * @param lines Corrected swath positions 
 * @return __global__ 
 */
__global__ void correctSwathsKernel(float headingFirst, 
                                    float headingDelta, 
                                    const Eigen::Vector4d* data,
                                    const int* bins,
                                    const float* binsFromCentre,
                                    float slantRes,
                                    int numPoints, 
                                    int numLines,
                                    Eigen::Vector2d* lines);

/**
 * @brief Calculates the swath positions (CUDA kernel)
 * 
 * @param yaw Yaw in radians
 * @param altitude Altitude in metres 
 * @param east UTM x coordinate
 * @param north UTM y coordinate 
 * @param slantRes Slant resolution
 * @param numSamples Number of samples 
 * @param bins Bin indices in swath
 * @param binsFromCentre Bin indices in swath from centre bin
 * @param line Corrected swath positions 
 * @return __global__ 
 */
__global__ void createSwathKernel(float yaw, 
                                  float altitude, 
                                  double east, 
                                  double north, 
                                  float slantRes,
                                  int numSamples,
                                  const int* bins,
                                  const float* binsFromCentre,
                                  Eigen::Vector2d* line);

/**
 * @brief Creates a mosaic from the swath positions and intensities (CUDA kernel)
 * 
 * @param minX UTM x coordinate of origin
 * @param minY UTM y coordinate of origin
 * @param resolution Mosaic resolution  
 * @param originalLines Original swath positions 
 * @param intensities Original swath intensities 
 * @param mask Water-column mask 
 * @param mosaic Mosaic image 
 * @param counter Number of swath points inside each mosaic cell
 * @param height Mosaic image height
 * @param width Mosaic image width 
 * @param numLines Number of pings
 * @param numPoints Number of samples 
 * @return __global__ 
 */
__global__ void createMosaicKernel(double minX, 
                                   double minY, 
                                   double resolution, 
                                   const Eigen::Vector2d* originalLines, 
                                   const int* intensities,
                                   const int* mask,
                                   int* mosaic,
                                   int* counter,
                                   int height,
                                   int width,
                                   int numLines,
                                   int numPoints);

/**
 * @brief Generates the corrected waterfall image (CUDA kernel)
 * 
 * @param mosaic Mosaic image 
 * @param count Number of swath points inside each mosaic cell
 * @param lines Corrected swath positions
 * @param height Mosaic image height
 * @param width Mosaic image width
 * @param numLines Number of pings
 * @param numPoints Number of samples
 * @param resolution Mosaic image resolution 
 * @param minX UTM x coordinate of origin
 * @param minY UTM y coordinate of origin
 * @param maxNeighbours Maximum number of neighbours for interpolation
 * @param waterfall Corrected waterfall image 
 * @return __global__ 
 */
__global__ void generateWaterfallKernel(const int* mosaic,
                                        const int* count,
                                        const Eigen::Vector2d* lines,
                                        int height,
                                        int width,
                                        int numLines,
                                        int numPoints,
                                        double resolution,
                                        double minX,
                                        double minY,
                                        int maxNeighbours,
                                        int* waterfall);
}
