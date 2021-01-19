#include <stdio.h>
#include <stdint.h>

//////////// Ultility ////////////
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
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

void readPnm(char *fileName, int &numChannels, int &width, int &height, uint8_t *&pixels)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
    {
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

    pixels = (uint8_t *)malloc(width * height * numChannels);
    for (int i = 0; i < width * height * numChannels; i++)
        fscanf(f, "%hhu", &pixels[i]);

    fclose(f);
}

void writePnm(const uint8_t *pixels, int numChannels, int width, int height, char *fileName)
{
    FILE *f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if (numChannels == 1)
        fprintf(f, "P2\n");
    else if (numChannels == 3)
        fprintf(f, "P3\n");
    else
    {
        fclose(f);
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height * numChannels; i++)
        fprintf(f, "%hhu\n", pixels[i]);

    fclose(f);
}

void writeEnergyMap(const char* filename, const uint32_t* energy_map, int width, int height)
{        
    FILE *f = fopen(filename, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)    
            fprintf(f, "%d ", energy_map[x + y * width]);        

        fprintf(f, "\n");
    }

    fclose(f);
}

float computeError(uint8_t *a1, uint8_t *a2, int n)
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

char *concatStr(const char *s1, const char *s2)
{
	char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
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
    switch (backend)
    {
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
    1, 0, -1,
};
const int ySobelFilter[] = {
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1
};

void hostApplyFilter(const uint8_t* inPixels, int width, int height, const int* filter, uint8_t* outPixels)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int sum = 0;
            for (int fY = 0; fY < FILTER_WIDTH; fY++)
            {
                for (int fX = 0; fX < FILTER_WIDTH; fX++)
                {
                    int xx = x + (fX - FILTER_WIDTH / 2);
                    int yy = y + (fY - FILTER_WIDTH / 2);
                    xx = min(max(xx, 0), width - 1);
                    yy = min(max(yy, 0), height - 1);
                    sum += inPixels[xx + yy * width] * filter[fX + fY * FILTER_WIDTH];
                }
            }
            
            outPixels[x + y * width] = max(min(sum, 255), 0);
        }
    }
}

void hostRemoveSeam(const uint8_t* inPixels, int width, int height, const uint32_t* energy_map, uint8_t* outPixels)
{
    // Find first min energy point
    int removeX = 0;
    for (int i = 1; i < width; i++)
    {
        if (energy_map[removeX] > energy_map[i])
            removeX = i;
    }

    for (int y = 0; y < height; y++)
    {
        // Copy pixels
        for (int x = 0; x < width - 1; x++)
        {
            int xx = x;
            if (x >= removeX)
                xx++;

            for (int channel = 0; channel < 3; channel++)             
                outPixels[(x + y * (width - 1)) * 3 + channel] = inPixels[(xx + y * width) * 3 + channel];            
        }

        // Find next min energy point
        if (y < height - 1)
        {
            int prevRX = removeX;
            for (int off = -1; off <= 1; off++)
            {
                int xx = prevRX + off;
                xx = min(xx, width - 1);
                xx = max(xx, 0);
                if (energy_map[removeX + (y + 1) * width] > energy_map[xx + (y + 1) * width])
                    removeX = xx;
            }
        }
    }
}

void hostHighlightSeam(const uint8_t* inPixels, int width, int height, const uint32_t* energy_map, uint8_t* outPixels)
{
    // Find first min energy point
    int removeX = 0;
    for (int i = 1; i < width; i++)
    {
        if (energy_map[removeX] > energy_map[i])
            removeX = i;
    }

    for (int y = 0; y < height; y++)
    {
        // Copy pixels
        for (int x = 0; x < width; x++)
        {
            int xx = x;
            if (xx == removeX)
            {
                outPixels[(x + y * width) * 3 + 0] = 255;
                outPixels[(x + y * width) * 3 + 1] = 0;
                outPixels[(x + y * width) * 3 + 2] = 0;
                continue;
            }

            for (int channel = 0; channel < 3; channel++)             
                outPixels[(x + y * width) * 3 + channel] = inPixels[(xx + y * width) * 3 + channel];            
        }

        // Find next min energy point
        if (y < height - 1)
        {
            int prevRX = removeX;
            for (int off = -1; off <= 1; off++)
            {
                int xx = prevRX + off;
                xx = min(xx, width - 1);
                xx = max(xx, 0);
                if (energy_map[removeX + (y + 1) * width] > energy_map[xx + (y + 1) * width])
                    removeX = xx;
            }
        }
    }
}

void hostSeamCarving(const uint8_t* inPixels, int width, int height, uint8_t* outPixels)
{
    uint8_t* gsPixels = (uint8_t*) malloc(sizeof(uint8_t) * width * height);

    // Convert to grayscale
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int i = y * width + x;
            uint8_t r = inPixels[3 * i];
            uint8_t g = inPixels[3 * i + 1];
            uint8_t b = inPixels[3 * i + 2];
            gsPixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    uint8_t* xEdges = (uint8_t*) malloc(sizeof(uint8_t) * width * height);
    hostApplyFilter(gsPixels, width, height, xSobelFilter, xEdges);

    uint8_t* yEdges = (uint8_t*) malloc(sizeof(uint8_t) * width * height);
    hostApplyFilter(gsPixels, width, height, ySobelFilter, yEdges);

    uint32_t* energy_map = (uint32_t*) malloc(sizeof(uint32_t) * width * height);
    for (int i = 0; i < width * height; i++)
        energy_map[i] = abs(xEdges[i]) + abs(yEdges[i]);

    for (int y = height - 2; y >= 0; y--)
    {
        for (int x = 0; x < width; x++)
        {       
            int minE = INT_MAX;
            for (int off = -1; off <= 1; off++)
            {
                int xx = x + off;
                xx = min(max(xx, 0), width - 1);
                minE = min(minE, energy_map[xx + (y + 1) * width]);
            }

            energy_map[x + y * width] += minE;
        }
    }

    writeEnergyMap("energy_map.txt", energy_map, width, height);

    // hostHighlightSeam(inPixels, width, height, energy_map, outPixels);
    // writePnm(outPixels, 3, width, height, "highlight.pnm");

    hostRemoveSeam(inPixels, width, height, energy_map, outPixels);

    free(gsPixels);
    free(xEdges);
    free(yEdges);
    free(energy_map);
}

const uint8_t* seamCarving(
    const uint8_t *inPixels,
    int width, int height,
    Backend backend,
    int outputWidth,
    dim3 blockSize = dim3(1)
) {
    GpuTimer timer;
    timer.Start();

    uint8_t* outPixels = (uint8_t*) malloc(sizeof(uint8_t) * width * height * 3);
    if (backend == Backend::Host) 
    {        
        uint8_t* bufferPixels = (uint8_t*) malloc(sizeof(uint8_t) * width * height * 3);
        memcpy(bufferPixels, inPixels, width * height * 3);
            
        uint8_t* inp = bufferPixels;
        uint8_t* out = outPixels;

        int carvedWidth = width - outputWidth;
        for (int i = 0; i < carvedWidth; i++)
        {
            hostSeamCarving(inp, width - i, height, out);            
            if (i < carvedWidth - 1)
            {
                uint8_t* temp = inp;
                inp = out;
                out = temp;
            }
        }

        outPixels = out;
        free(inp);
    }
    else
    {

    }

	timer.Stop();
	float time = timer.Elapsed();
    printf("Processing time (use %s): %f ms\n\n", strEnum(backend), time);

    return outPixels;
}


int main(int argc, char ** argv)
{
    printDeviceInfo();

    char* inputFile;
    if (argc >= 2)
    {
        inputFile = argv[1];
    }
    else 
    {
        printf("File name is required");
        return 1;
    }

    int outWidth;
    if (argc >= 3)
    {
        outWidth = atoi(argv[2]);
    }
    else
    {
        printf("Output width is required");
        return 1;
    }

    int operatingMode = Backend::Host | Backend::Kernel1 | Backend::Kernel2;
    if (argc >= 4)
        operatingMode = atoi(argv[3]);
    
    int blockSize = 128;
    if (argc >= 5)
        blockSize = atoi(argv[4]);

    int numChannels, width, height;
    uint8_t* inPixels;
    readPnm(inputFile, numChannels, width, height, inPixels);

    if (numChannels != 3)
    {
        printf("Only RGB image is supported");
        return 1;
    }

    if (outWidth > width - 1)
    {
        printf("Output width is too big, maximum output width is: %d", width - 1);
        return 1;
    }

    const uint8_t* hostOutPixels = seamCarving(inPixels, width, height, Backend::Host, outWidth);
    // const uint8_t* k1OutPixels = seamCarving(inPixels, width, height, Backend::Kernel1, outWidth);
    
	char* outFileNameBase = strtok(inputFile, "."); // Get rid of extension
	writePnm(hostOutPixels, 3, outWidth, height, concatStr(outFileNameBase, "_host.pnm"));
    // writePnm(k1OutPixels, 3, outWidth, height, concatStr(outFileNameBase, "_device1.pnm"));
    
    free((void*) hostOutPixels);
    
    return 0;
}
