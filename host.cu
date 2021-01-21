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
                int xx = min(max(prevRX + off, 0), width - 1);
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
                int xx = min(max(prevRX + off, 0), width - 1);
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
                int xx = min(max(x + off, 0), width - 1);
                minE = min(minE, energyMap[xx + (y + 1) * width]);
            }

            energyMap[x + y * width] += minE;
        }
    }

#ifdef WRITE_LOG
    writeEnergyMap("energy_map.txt", energyMap, width, height);
#endif
}

uint8_t* seamCarving(const uint8_t* inPixels, int width, int height, int outputWidth)
{
    int gsPixelSize = width * height;
    int rgbPixelSize = width * height * 3;
    int energySize = sizeof(uint32_t) * width * height;

    GpuTimer timer;
    timer.Start();

    uint8_t* outPixels = (uint8_t*)malloc(width * height * 3);
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

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time: %f ms\n\n", time);

    return outPixels;
}

int main(int argc, char** argv)
{
    // printDeviceInfo();

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

    uint8_t* hostOutPixels = seamCarving(inPixels, width, height, outWidth);
    char* outFileNameBase = strtok(inputFile, "."); // Get rid of extension
    writePnm(hostOutPixels, 3, outWidth, height, concatStr(outFileNameBase, "_host.pnm"));

    free(hostOutPixels);

    return 0;
}
