
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda_gl_interop.h>

#include <iostream>

#define GL_TEXTURE_2D 0x0DE1

static cudaGraphicsResource *cudaResource;
cudaArray_t cuArray;
unsigned char* oldBuffer;
unsigned char* newBuffer;

// kernel
__device__ unsigned int wang_hash(unsigned int seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}

__device__ int findNeighbours(int x, int y, unsigned char* buffer, int w, int h) {
	if (x >= w || y >= h) return;
	int n = 0;
	for (int i = y - 1; i <= y + 1; i++) {
		for (int j = x - 1; j <= x + 1; j++) {
			if (i < 0 || i >= h || j < 0 || j >= w) continue; // bounds
			if (i == y && j == x) continue; // skip self
			int idx = i * w + j;
			if (buffer[idx] > 0) n++;
		}
	}
	return n;
}

__global__ void updateGrid(unsigned char* buffer, unsigned char* output,int w, int h) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int idx = y * w + x;

	int n = findNeighbours(x, y, buffer, w, h);

	int value = buffer[idx];

	if (value > 0) { // currently alive
		if (n < 2 || n > 3) value = 0; // die
	}
	else { // currently dead
		if (n == 3) value = 255; // birth
	}

	output[idx] = value;
}

__global__ void generateRandomGrid(unsigned char* destination, int w, int h, int seed) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int idx = y * w + x;
	int hash = wang_hash(seed ^ x * y);
	int value = (hash & 1) ? 255 : 0;
	destination[idx] = value;
}

__global__ void bufferToTexture(unsigned char* source, cudaSurfaceObject_t destination, int w, int h) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int idx = y * w + x;

	uchar1 pixel = make_uchar1(source[idx]);
	surf2Dwrite(pixel, destination, x * sizeof(uchar1), y);
}

// helpers
cudaSurfaceObject_t mapCudaResource() {
	cudaGraphicsMapResources(1, &cudaResource, 0);
	cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaSurfaceObject_t surface = 0;
	cudaCreateSurfaceObject(&surface, &resDesc);

	return surface;
}

void unmapCudaResource(cudaSurfaceObject_t surface) {
	cudaDestroySurfaceObject(surface);
	cudaGraphicsUnmapResources(1, &cudaResource, 0);
}

// external functions
extern "C" __declspec(dllexport)
void cudaRegisterTexture(GLuint texture) {
	cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

extern "C" __declspec(dllexport)
void cudaAllocateBuffer(int w, int h) {
	cudaMalloc(&oldBuffer, w * h * sizeof(unsigned char));
	cudaMalloc(&newBuffer,w*h*sizeof(unsigned char));
}

extern "C" __declspec(dllexport)
void cudaFreeBuffer() {
	cudaFree(&oldBuffer);
	cudaFree(&newBuffer);
}

extern "C" __declspec(dllexport)
void cudaFillRandom(int w, int h, int seed) {
	cudaSurfaceObject_t surface = mapCudaResource();

	dim3 block(16, 16);
	dim3 grid((w + 15) / 16, (h + 15) / 16);
	generateRandomGrid << <grid, block >> > (oldBuffer, w, h,seed);
	cudaDeviceSynchronize();
	bufferToTexture << <grid, block >> > (oldBuffer, surface,w,h);
	cudaDeviceSynchronize();

	unmapCudaResource(surface);
}

extern "C" __declspec(dllexport)
void cudaUpdateTexture(int w, int h) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	dim3 block(16, 16);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	updateGrid << <grid, block >> > (oldBuffer, newBuffer, w, h);
	cudaDeviceSynchronize();

	cudaSurfaceObject_t surface = mapCudaResource();
	bufferToTexture << <grid, block >> > (newBuffer, surface, w,h);
	cudaDeviceSynchronize();
	unmapCudaResource(surface);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Kernel execution time: " << milliseconds << " ms\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	unsigned char* tmp = oldBuffer;
	oldBuffer = newBuffer;
	newBuffer = tmp;
}
