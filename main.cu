#include <stdint.h>
#include <stdio.h>

//////////// Ultility ////////////
#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess) {                                \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                cudaGetErrorString(error));                        \
            exit(1);                                               \
        }                                                          \
    }

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char* fileName, int& numChannels, int& width, int& height, uint8_t*& pixels)
{
    FILE* f = fopen(fileName, "r");
    if (f == NULL) {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);
    if (strcmp(type, "P2") == 0)
        numChannels = 1;
    else if (strcmp(type, "P3") == 0)
        numChannels = 3;
    else // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uint8_t*)malloc(width * height * numChannels);
    for (int i = 0; i < width * height * numChannels; i++)
        fscanf(f, "%hhu", &pixels[i]);

    fclose(f);
}

void writePnm(const uint8_t* pixels, int numChannels, int width, int height, char* fileName)
{
    FILE* f = fopen(fileName, "w");
    if (f == NULL) {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if (numChannels == 1)
        fprintf(f, "P2\n");
    else if (numChannels == 3)
        fprintf(f, "P3\n");
    else {
        fclose(f);
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height * numChannels; i++)
        fprintf(f, "%hhu\n", pixels[i]);

    fclose(f);
}

void writeEnergyMap(const char* filename, const uint32_t* energyMap, int width, int height)
{
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        printf("Cannot write %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            fprintf(f, "%d ", energyMap[x + y * width]);

        fprintf(f, "\n");
    }

    fclose(f);
}

float computeError(uint8_t* a1, uint8_t* a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
        err += abs((int)a1[i] - (int)a2[i]);
    err /= n;
    return err;
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

char* concatStr(const char* s1, const char* s2)
{
    char* result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

enum Backend {
    Host = 1,
    Kernel1 = 2,
    Kernel2 = 4,
};

const char* strEnum(Backend backend)
{
    switch (backend) {
    case Backend::Host:
        return "host";
    case Backend::Kernel1:
        return "device kernel 1";
    case Backend::Kernel2:
        return "device kernel 1";
    }

    return "unknown";
}

//////////// Implementations ////////////
#define FILTER_WIDTH 3

const int xSobelFilter[] = {
    1, 0, -1,
    2, 0, -2,
    1, 0, -1
};

const int ySobelFilter[] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1
};

__constant__ int dc_xSobelFilter[FILTER_WIDTH * FILTER_WIDTH];
__constant__ int dc_ySobelFilter[FILTER_WIDTH * FILTER_WIDTH];

/// HOST ///
void hostGrayscale(const uint8_t* inPixels, int width, int height, uint8_t* outPixels)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            uint8_t r = inPixels[3 * i];
            uint8_t g = inPixels[3 * i + 1];
            uint8_t b = inPixels[3 * i + 2];
            outPixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}

void hostRemoveSeam(const uint8_t* inPixels, int width, int height, int channels, const uint32_t* energyMap, uint8_t* outPixels)
{
    // Find first min energy point
    int removeX = 0;
    for (int i = 1; i < width; i++) {
        if (energyMap[removeX] > energyMap[i])
            removeX = i;
    }

    for (int y = 0; y < height; y++) {
        // Copy pixels
        for (int x = 0; x < width - 1; x++) {
            int xx = x;
            if (x >= removeX)
                xx++;

            for (int c = 0; c < channels; c++)
                outPixels[(x + y * (width - 1)) * channels + c] = inPixels[(xx + y * width) * channels + c];
        }

        // Find next min energy point
        if (y < height - 1) {
            int prevRX = removeX;
            for (int off = -1; off <= 1; off++) {
                int xx = prevRX + off;
                xx = min(xx, width - 1);
                xx = max(xx, 0);
                if (energyMap[removeX + (y + 1) * width] > energyMap[xx + (y + 1) * width])
                    removeX = xx;
            }
        }
    }
}

void hostHighlightSeam(const uint8_t* inPixels, int width, int height, const uint32_t* energyMap, uint8_t* outPixels)
{
    // Find first min energy point
    int removeX = 0;
    for (int i = 1; i < width; i++) {
        if (energyMap[removeX] > energyMap[i])
            removeX = i;
    }

    for (int y = 0; y < height; y++) {
        // Copy pixels
        for (int x = 0; x < width; x++) {
            int xx = x;
            if (xx == removeX) {
                outPixels[(x + y * width) * 3 + 0] = 255;
                outPixels[(x + y * width) * 3 + 1] = 0;
                outPixels[(x + y * width) * 3 + 2] = 0;
                continue;
            }

            for (int channel = 0; channel < 3; channel++)
                outPixels[(x + y * width) * 3 + channel] = inPixels[(xx + y * width) * 3 + channel];
        }

        // Find next min energy point
        if (y < height - 1) {
            int prevRX = removeX;
            for (int off = -1; off <= 1; off++) {
                int xx = prevRX + off;
                xx = min(xx, width - 1);
                xx = max(xx, 0);
                if (energyMap[removeX + (y + 1) * width] > energyMap[xx + (y + 1) * width])
                    removeX = xx;
            }
        }
    }
}

void hostFindEnergyMap(const uint8_t* gsInPixels, int width, int height, uint32_t* energyMap)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int xEdge = 0;
            int yEdge = 0;
            for (int fY = 0; fY < FILTER_WIDTH; fY++) {
                for (int fX = 0; fX < FILTER_WIDTH; fX++) {
                    int xx = x + (fX - FILTER_WIDTH / 2);
                    int yy = y + (fY - FILTER_WIDTH / 2);
                    xx = min(max(xx, 0), width - 1);
                    yy = min(max(yy, 0), height - 1);
                    xEdge += gsInPixels[xx + yy * width] * xSobelFilter[fX + fY * FILTER_WIDTH];
                    yEdge += gsInPixels[xx + yy * width] * ySobelFilter[fX + fY * FILTER_WIDTH];
                }
            }

            xEdge = max(min(xEdge, 255), 0);
            yEdge = max(min(yEdge, 255), 0);
            energyMap[x + y * width] = xEdge + yEdge;
        }
    }

#ifdef WRITE_LOG
    writeEnergyMap("energy.txt", energyMap, width, height);
#endif

    for (int y = height - 2; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int minE = INT_MAX;
            for (int off = -1; off <= 1; off++) {
                int xx = x + off;
                xx = min(max(xx, 0), width - 1);
                minE = min(minE, energyMap[xx + (y + 1) * width]);
            }

            energyMap[x + y * width] += minE;
        }
    }

#ifdef WRITE_LOG
    writeEnergyMap("energy_map.txt", energyMap, width, height);
#endif
}

/// DEVICE ///

__global__ void deviceGrayscale(const uint8_t* inPixels, int width, int height, uint8_t* gsOutPixels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int offset = x + y * width;
        gsOutPixels[offset] = uint8_t(
            float(inPixels[offset * 3 + 0]) * 0.299f + float(inPixels[offset * 3 + 1]) * 0.587f + float(inPixels[offset * 3 + 2]) * 0.114f);
    }
}

__global__ void deviceCalcEnergy(const uint8_t* gsInPixels, int width, int height, uint32_t* outEnergy)
{
    extern __shared__ uint8_t s_inPixels[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blkDimX = blockDim.x;
    int blkDimY = blockDim.y;
    int offsetX = blockIdx.x * blkDimX;
    int offsetY = blockIdx.y * blkDimY;
    int x = tx + offsetX;
    int y = ty + offsetY;

    if (x >= width || y >= height)
        return;

    int padding = FILTER_WIDTH / 2;
    int boundWidth = blkDimX + FILTER_WIDTH - 1;
    int boundHeight = blkDimY + FILTER_WIDTH - 1;

    for (int j = ty; j < boundHeight; j += blkDimY) {
        for (int i = tx; i < boundWidth; i += blkDimX) {
            int xx = min(max(i + offsetX - padding, 0), width - 1);
            int yy = min(max(j + offsetY - padding, 0), height - 1);
            s_inPixels[i + j * boundWidth] = gsInPixels[xx + yy * width];
        }
    }

    __syncthreads();

    int xEdge = 0;
    int yEdge = 0;
    for (int j = 0; j < FILTER_WIDTH; j++) {
        for (int i = 0; i < FILTER_WIDTH; i++) {
            int xx = tx + i;
            int yy = ty + j;
            int filterIdx = i + j * FILTER_WIDTH;

            int pixelVal = s_inPixels[xx + yy * boundWidth];
            xEdge += pixelVal * dc_xSobelFilter[filterIdx];
            yEdge += pixelVal * dc_ySobelFilter[filterIdx];
        }
    }

    xEdge = max(min(xEdge, 255), 0);
    yEdge = max(min(yEdge, 255), 0);
    outEnergy[x + y * width] = xEdge + yEdge;
}

// FIXME: If image width > blockDim, a.k.a we run on more than 1 grid
//       the seam would never be able to cross between block border
__global__ void deviceFindEnergyMap(const uint32_t* inEnergy, int width, int height, uint32_t* outEnergyMap)
{
    extern __shared__ uint32_t s_rowEnergy[];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width)
        return;

    int idx = x + (height - 1) * width;
    uint32_t energy = inEnergy[idx];

    s_rowEnergy[threadIdx.x] = energy;
    outEnergyMap[idx] = energy;

    __syncthreads();

    int tx = threadIdx.x;
    for (int y = height - 2; y >= 0; y--) {
        int left = tx == 0 ? INT_MAX : s_rowEnergy[tx - 1];
        int middle = s_rowEnergy[tx];
        int right = INT_MAX;
        if (tx < width - 1 && tx < blockDim.x - 1)
            right = s_rowEnergy[tx + 1];

        idx = x + y * width;
        uint32_t minimum = min(middle, min(left, right));
        uint32_t energy = inEnergy[idx] + minimum;

        __syncthreads();
        s_rowEnergy[tx] = energy;
        outEnergyMap[idx] = energy;
    }
}

uint8_t* seamCarving(
    const uint8_t* inPixels,
    int width, int height,
    Backend backend,
    int outputWidth,
    dim3 blockSize = dim3(1, 1))
{
    int gsPixelSize = width * height;
    int rgbPixelSize = width * height * 3;
    int energySize = sizeof(uint32_t) * width * height;

    GpuTimer timer;
    timer.Start();

    uint8_t* outPixels = (uint8_t*)malloc(width * height * 3);
    if (backend == Backend::Host) {
        uint8_t* gsInpPixels = (uint8_t*)malloc(gsPixelSize);
        uint8_t* gsOutPixels = (uint8_t*)malloc(gsPixelSize);
        uint8_t* bufferRGBPixels = (uint8_t*)malloc(rgbPixelSize);
        uint32_t* energyMap = (uint32_t*)malloc(energySize);

        memcpy(bufferRGBPixels, inPixels, rgbPixelSize);

        hostGrayscale(inPixels, width, height, gsInpPixels);

#ifdef WRITE_LOG
        writePnm(gsInpPixels, 1, width, height, "grey.pnm");
#endif

        uint8_t* gsInp = gsInpPixels;
        uint8_t* gsOut = gsOutPixels;
        uint8_t* rgbInp = bufferRGBPixels;
        uint8_t* rgbOut = outPixels;

        int carvedWidth = width - outputWidth;
        for (int i = 0; i < carvedWidth; i++) {
            hostFindEnergyMap(gsInp, width - i, height, energyMap);

#ifdef WRITE_LOG
            hostHighlightSeam(rgbInp, width - i, height, energyMap, rgbOut);
            writePnm(rgbOut, 3, width - i, height, "highlight.pnm");
#endif

            hostRemoveSeam(gsInp, width - i, height, 1, energyMap, gsOut);
            hostRemoveSeam(rgbInp, width - i, height, 3, energyMap, rgbOut);

            if (i < carvedWidth - 1) {
                uint8_t* temp = gsInp;
                gsInp = gsOut;
                gsOut = temp;

                temp = rgbInp;
                rgbInp = rgbOut;
                rgbOut = temp;
            }
        }

        outPixels = rgbOut;

        free(gsInp);
        free(gsOut);
        free(rgbInp);
        free(energyMap);
    } else {
        uint8_t* gsInpPixels = (uint8_t*)malloc(gsPixelSize);
        uint8_t* gsOutPixels = (uint8_t*)malloc(gsPixelSize);
        uint8_t* bufferRGBPixels = (uint8_t*)malloc(rgbPixelSize);
        uint32_t* energyMap = (uint32_t*)malloc(energySize);

        memcpy(bufferRGBPixels, inPixels, rgbPixelSize);

        uint8_t* d_gsInpPixels;
        uint8_t* d_rgbInpPixels;
        uint32_t* d_energy;
        uint32_t* d_energy_map;

        CHECK(cudaMalloc(&d_gsInpPixels, gsPixelSize));
        CHECK(cudaMalloc(&d_rgbInpPixels, rgbPixelSize));
        CHECK(cudaMalloc(&d_energy, energySize));
        CHECK(cudaMalloc(&d_energy_map, energySize));

        CHECK(cudaMemcpyToSymbol(dc_xSobelFilter, xSobelFilter, FILTER_WIDTH * FILTER_WIDTH * sizeof(int), 0, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyToSymbol(dc_ySobelFilter, ySobelFilter, FILTER_WIDTH * FILTER_WIDTH * sizeof(int), 0, cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(d_rgbInpPixels, inPixels, rgbPixelSize, cudaMemcpyHostToDevice));

        dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
        deviceGrayscale<<<gridSize, blockSize>>>(d_rgbInpPixels, width, height, d_gsInpPixels);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(gsInpPixels, d_gsInpPixels, gsPixelSize, cudaMemcpyDeviceToHost));

#ifdef WRITE_LOG
        writePnm(gsInpPixels, 1, width, height, "grey_dev.pnm");
#endif

        uint8_t* gsInp = gsInpPixels;
        uint8_t* gsOut = gsOutPixels;
        uint8_t* rgbInp = bufferRGBPixels;
        uint8_t* rgbOut = outPixels;

        int carvedWidth = width - outputWidth;
        for (int i = 0; i < carvedWidth; i++) {

            int eWidth = width - i;

            // Calculate energy: E = xSobel + ySobel
            dim3 gridSize((eWidth - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
            int sMemSize = (blockSize.x + FILTER_WIDTH - 1) * (blockSize.y + FILTER_WIDTH - 1) * sizeof(uint8_t);
            deviceCalcEnergy<<<gridSize, blockSize, sMemSize>>>(d_gsInpPixels, eWidth, height, d_energy);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());

#ifdef WRITE_LOG
            CHECK(cudaMemcpy(energyMap, d_energy, energySize, cudaMemcpyDeviceToHost));
            writeEnergyMap("energy_dev.txt", energyMap, eWidth, height);
#endif

            // Find energy map line by line
            int fmBlockSize = blockSize.x * blockSize.y;
            int fmGridSize = (eWidth - 1) / fmBlockSize + 1;
            int fmSMemSize = fmBlockSize * sizeof(uint32_t);
            deviceFindEnergyMap<<<fmGridSize, fmBlockSize, fmSMemSize>>>(d_energy, eWidth, height, d_energy_map);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());

            // Remove seam on host based on energy map
            CHECK(cudaMemcpy(energyMap, d_energy_map, energySize, cudaMemcpyDeviceToHost));
#ifdef WRITE_LOG
            writeEnergyMap("energy_map_dev.txt", energyMap, eWidth, height);
            hostHighlightSeam(rgbInp, eWidth, height, energyMap, rgbOut);
            writePnm(rgbOut, 3, eWidth, height, "highlight_dev.pnm");
#endif
            hostRemoveSeam(gsInp, eWidth, height, 1, energyMap, gsOut);
            hostRemoveSeam(rgbInp, eWidth, height, 3, energyMap, rgbOut);

            if (i < carvedWidth - 1) {
                uint8_t* temp = gsInp;
                gsInp = gsOut;
                gsOut = temp;

                temp = rgbInp;
                rgbInp = rgbOut;
                rgbOut = temp;

                CHECK(cudaMemcpy(d_gsInpPixels, gsInp, gsPixelSize, cudaMemcpyHostToDevice));
            }
        }

        outPixels = rgbOut;

        free(gsInp);
        free(gsOut);
        free(rgbInp);
        free(energyMap);

        CHECK(cudaFree(d_gsInpPixels));
        CHECK(cudaFree(d_rgbInpPixels));
        CHECK(cudaFree(d_energy));
        CHECK(cudaFree(d_energy_map));
    }

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use %s): %f ms\n\n", strEnum(backend), time);

    return outPixels;
}

int main(int argc, char** argv)
{
    printDeviceInfo();

    char* inputFile;
    if (argc >= 2) {
        inputFile = argv[1];
    } else {
        printf("File name is required");
        return 1;
    }

    int outWidth;
    if (argc >= 3) {
        outWidth = atoi(argv[2]);
    } else {
        printf("Output width is required");
        return 1;
    }

    int operatingMode = Backend::Host | Backend::Kernel1 | Backend::Kernel2;
    if (argc >= 4)
        operatingMode = atoi(argv[3]);

    dim3 blockSize(32, 32);
    if (argc >= 5) {
        int v = atoi(argv[4]);
        blockSize.x = v;
        blockSize.y = v;
    }

    int numChannels, width, height;
    uint8_t* inPixels;
    readPnm(inputFile, numChannels, width, height, inPixels);

    if (numChannels != 3) {
        printf("Only RGB image is supported");
        return 1;
    }

    if (outWidth > width - 1) {
        printf("Output width is too big, maximum output width is: %d", width - 1);
        return 1;
    }

    uint8_t* hostOutPixels = seamCarving(inPixels, width, height, Backend::Host, outWidth, blockSize);
    uint8_t* k1OutPixels = seamCarving(inPixels, width, height, Backend::Kernel1, outWidth, blockSize);

    char* outFileNameBase = strtok(inputFile, "."); // Get rid of extension
    writePnm(hostOutPixels, 3, outWidth, height, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(k1OutPixels, 3, outWidth, height, concatStr(outFileNameBase, "_device1.pnm"));

    float err = computeError(hostOutPixels, k1OutPixels, outWidth * height * 3);
    printf("Error: %f\n", err);

    free(hostOutPixels);
    free(k1OutPixels);

    return 0;
}
