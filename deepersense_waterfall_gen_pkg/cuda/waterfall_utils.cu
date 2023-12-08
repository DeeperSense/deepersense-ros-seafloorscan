#include "waterfall_utils.hpp"

__global__ void utils::createSwathKernel(float yaw, 
                                         float altitude, 
                                         double east, 
                                         double north, 
                                         float slantRes,
                                         int numSamples,
                                         const int* bins,
                                         const float* binsFromCentre,
                                         Eigen::Vector2d* line)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numSamples)
    {    
        int numBlindBins = roundf(altitude / slantRes);
        int numGroundBins = numSamples / 2 - numBlindBins;
    
        bool blind = (numGroundBins <= bins[i]) && (bins[i] < numGroundBins + 2 * numBlindBins);
        float x = 0.0;
    
        if (!blind)
        {
            float sign = 0.0;
            if (binsFromCentre[i] < 0.0)
                sign = -1.0; 
            else if (binsFromCentre[i] > 0.0)
                sign = 1.0;
    
            x = sqrtf(powf(slantRes * binsFromCentre[i], 2) - powf(altitude, 2)) * sign;
        }

        line[i].x() = cos(yaw) * x + east;
        line[i].y() = sin(yaw) * x + north;
    }
}

void utils::createSwath(float yaw, 
                       float altitude, 
                       double east, 
                       double north, 
                       float slantRes, 
                       int numSamples, 
                       const std::vector<int>& swathBins, 
                       const std::vector<float>& swathBinsFromCentre, 
                       std::vector<Eigen::Vector2d>& line)
{
    int* bins;
    cudaMalloc((void **)&bins, numSamples * sizeof(int));
    cudaMemcpy(bins, swathBins.data(), numSamples * sizeof(int), cudaMemcpyHostToDevice);

    float* binsFromCentre;
    cudaMalloc((void **)&binsFromCentre, numSamples * sizeof(float));
    cudaMemcpy(binsFromCentre, swathBinsFromCentre.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice);

    Eigen::Vector2d* original;
    cudaMalloc((void **)&original, numSamples * sizeof(Eigen::Vector2d));

    int numBlocks = (numSamples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    createSwathKernel<<<numBlocks, BLOCK_SIZE>>>(yaw, altitude, east, north, slantRes, numSamples, bins, binsFromCentre, original);

    cudaDeviceSynchronize();
    cudaMemcpy(line.data(), original, numSamples * sizeof(Eigen::Vector2d), cudaMemcpyDeviceToHost);
   
    cudaFree(bins);
    cudaFree(binsFromCentre);
    cudaFree(original);
}

__global__ void utils::correctSwathsKernel(float headingFirst, 
                                           float headingDelta, 
                                           const Eigen::Vector4d* data,
                                           const int* bins,
                                           const float* binsFromCentre,
                                           float slantRes,
                                           int numPoints, 
                                           int numLines,
                                           Eigen::Vector2d* lines)
{
    // data (x, y, altitude, yaw)
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numPoints && j < numLines)
    {
        float yaw = headingFirst + j * headingDelta;
    
        int numBlindBins = roundf(data[j](2) / slantRes);
        int numGroundBins = numPoints / 2 - numBlindBins;
    
        bool blind = (numGroundBins <= bins[i]) && (bins[i] < numGroundBins + 2 * numBlindBins);
        float x = 0.0;
    
        if (!blind)
        {
            float sign = 0.0;
            if (binsFromCentre[i] < 0.0)
                sign = -1.0; 
            else if (binsFromCentre[i] > 0.0)
                sign = 1.0;
    
            x = sqrtf(powf(slantRes * binsFromCentre[i], 2) - powf(data[j](2), 2)) * sign;
        }    

        lines[j * numPoints + i].x() = cos(yaw) * x + data[j](0);
        lines[j * numPoints + i].y() = sin(yaw) * x + data[j](1);
    }
}

__global__ void utils::createMosaicKernel(double minX, 
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
                                          int numPoints)
{
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (tidY < numLines) && (tidX < numPoints) )
    {
        Eigen::Vector2d point = originalLines[tidY * numPoints + tidX];

        // calculate pixel position
        int x = double2int( roundf((point.x() - minX) / resolution)) ;
        int y = double2int( roundf((point.y() - minY) / resolution)) ;

        x = max(0, x);
        y = max(0, y);
        x = min(width - 1, x);
        y = min(height - 1, y);
        
        int idx = width * y + x;

        if (mask[tidY * numPoints + tidX])
        {
            atomicAdd(&mosaic[idx], intensities[tidY * numPoints + tidX]);
            atomicAdd(&counter[idx], 1);
        }
    }
}

__global__ void utils::generateWaterfallKernel(const int* mosaic,
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
                                               int* waterfall)
{
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (tidY < numLines) && (tidX < numPoints) )
    {
        Eigen::Vector2d point = lines[tidY * numPoints + tidX];

        // calculate pixel position
        int x = (point.x() - minX) / resolution;
        int y = (point.y() - minY) / resolution;

        x = max(0, x);
        y = max(0, y);
        x = min(width - 1, x);
        y = min(height - 1, y);

        int idx = width * y + x;

        if (count[idx] >= 1)
        {
            waterfall[tidY * numPoints + tidX] = mosaic[idx] / count[idx];
        }
        else
        {
            int goodNeighbours = 0;
            int radius = 1;
            int sumIntensities = 0;

            while (true)
            {
                int minY = max(0, y - radius);
                int minX = max(0, x - radius);
                int maxY = min(height - 1, y + radius);
                int maxX = min(width - 1, x + radius);

                for (int i = minX; i != maxX + 1; ++i)
                {
                    if (count[minY * width + i] > 0)
                    {
                        sumIntensities += mosaic[minY * width + i] / int2float(count[minY * width + i]);
                        ++goodNeighbours;

                        if (goodNeighbours >= maxNeighbours)
                            break;
                    }
                    
                    if (count[maxY * width + i] > 0)
                    {
                        sumIntensities += mosaic[maxY * width + i] / int2float(count[maxY * width + i]);
                        ++goodNeighbours;

                        if (goodNeighbours >= maxNeighbours)
                            break;
                    }
                }

                if (goodNeighbours >= maxNeighbours)
                    break;

                for (int i = minY; i != maxY + 1; ++i)
                {
                    if (count[i*width + minX] > 0)
                    {
                        sumIntensities += mosaic[i*width + minX] / int2float(count[i * width + minX]);
                        ++goodNeighbours;

                        if (goodNeighbours >= maxNeighbours)
                            break;
                    }
                    if (count[i*width + maxX] > 0)
                    {
                        sumIntensities += mosaic[i*width + maxX] / int2float(count[i * width + maxX]);
                        ++goodNeighbours;

                        if (goodNeighbours >= maxNeighbours)
                            break;
                    }
                }

                if (goodNeighbours >= maxNeighbours)
                    break;

                ++radius;
            }
            
            waterfall[tidY * numPoints + tidX] = sumIntensities / goodNeighbours;
        }
    }
}

void utils::interpolate(int numLines, 
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
                        std::vector<int>& correctedIntensities)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dim3 blockDim32(32, 32);
    dim3 gridDim32Lines((numPoints + blockDim32.x - 1) / blockDim32.x, (numLines + blockDim32.y - 1) / blockDim32.y);

    // load all data in the CUDA memory

    Eigen::Vector4d* data;
    cudaMalloc((void **)&data, numLines * sizeof(Eigen::Vector4d));
    cudaMemcpy(data, swathData.data(), numLines * sizeof(Eigen::Vector4d), cudaMemcpyHostToDevice);

    int* bins;
    cudaMalloc((void **)&bins, numPoints * sizeof(int));
    cudaMemcpy(bins, swathBins.data(), numPoints * sizeof(int), cudaMemcpyHostToDevice);

    float* binsFromCentre;
    cudaMalloc((void **)&binsFromCentre, numPoints * sizeof(float));
    cudaMemcpy(binsFromCentre, swathBinsFromCentre.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice);

    Eigen::Vector2d* original;
    cudaMalloc((void **)&original, numLines * numPoints * sizeof(Eigen::Vector2d));
    cudaMemcpy(original, originalLines.data(), numPoints * numLines * sizeof(Eigen::Vector2d), cudaMemcpyHostToDevice);

    int* columnMask;
    cudaMalloc((void **)&columnMask, numLines * numPoints * sizeof(int));
    cudaMemcpy(columnMask, waterColumnMask.data(), numPoints * numLines * sizeof(int), cudaMemcpyHostToDevice);

    Eigen::Vector2d* corrected;
    cudaMallocHost((void **)&corrected, numLines * numPoints * sizeof(Eigen::Vector2d));

    int* intensities;
    cudaMalloc((void **)&intensities, numLines * numPoints * sizeof(int));
    cudaMemcpy(intensities, originalIntensities.data(),  numLines * numPoints * sizeof(int), cudaMemcpyHostToDevice);

    int* waterfall;
    cudaMallocHost((void **)&waterfall, numLines * numPoints * sizeof(int));

    float headingStart = swathData[0](3);
    float delta = (swathData[numLines-1](3) - swathData[0](3)) / (numLines - 1);  

    // correct swaths  

    correctSwathsKernel<<<gridDim32Lines, blockDim32>>>(headingStart, delta, data, bins,
                                                binsFromCentre, slantRes, numPoints, numLines, 
                                                corrected);
    
    cudaDeviceSynchronize();
    cudaFree(bins);
    cudaFree(binsFromCentre);
    cudaFree(data);

    // calculate minimum and maximum UTM coordinates 

    Eigen::Vector2d minimum, maximum, minimum2, maximum2;
    findMinMax<BLOCK_SIZE, double>(original, numLines * numPoints, minimum, maximum);
    findMinMax<BLOCK_SIZE, double>(corrected, numLines * numPoints, minimum2, maximum2);

    Eigen::Vector2d overallMin, overallMax;
    overallMin << (double) min(minimum.x(), minimum2.x()), min(minimum.y(), minimum2.y());
    overallMax << (double) max(maximum.x(), maximum2.x()), max(maximum.y(), maximum2.y());

    // create mosaic  

    int height = (int) ceil( (overallMax(1) - overallMin(1)) / resolution);
    int width = (int) ceil( (overallMax(0) - overallMin(0)) / resolution);

    std::vector<int> grid(width * height, 0);
    std::vector<int> counter(width * height, 0);

    int* count;
    cudaMalloc((void **)&count, height * width * sizeof(int));
    cudaMemcpy(count, counter.data(),  height * width * sizeof(int), cudaMemcpyHostToDevice);

    int* mosaic;
    cudaMalloc((void **)&mosaic, height * width * sizeof(int));
    cudaMemcpy(mosaic, grid.data(),  height * width * sizeof(int), cudaMemcpyHostToDevice);

    createMosaicKernel<<<gridDim32Lines, blockDim32>>>(overallMin.x(), overallMin.y(), (double) resolution, 
                                            original, intensities, columnMask, mosaic,
                                            count, height, width, numLines, numPoints);
    
    cudaDeviceSynchronize();
    cudaFree(intensities);
    cudaFree(original);
    cudaFree(columnMask);

    // generate corrected waterfall 
    
    generateWaterfallKernel<<<gridDim32Lines, blockDim32>>>(mosaic, count, corrected, height, width,
                                                    numLines, numPoints, resolution, overallMin.x(), overallMin.y(),
                                                    maxNeighbours, waterfall);

    cudaDeviceSynchronize();

    // copy results back to CPU 

    cudaMemcpy(correctedIntensities.data(), waterfall, numLines * numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(correctedLines.data(), corrected, numLines * numPoints * sizeof(Eigen::Vector2d), cudaMemcpyDeviceToHost);

    cudaFree(mosaic);
    cudaFree(count);
    cudaFree(corrected);
    cudaFree(waterfall);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Waterfall correction: Time elapsed: %f ms.\n", milliseconds);
}
