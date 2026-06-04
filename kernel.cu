#include "kernel.cuh"

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

__global__ void updateGrid(unsigned char* buffer, unsigned char* output, cudaSurfaceObject_t destination, int w, int h) {
	__shared__ unsigned char sData[18 * 18];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;
	if (x >= w || y >= h) return;
	int idx = y * w + x;

	// load into shared
	int sx = tx + 1;
	int sy = ty + 1;

	// 1) center
	sData[sy * 18 + sx] = buffer[idx];

	// ---- HALO ----

	// left/right
	if (tx == 0) {
		int lx = x - 1;
		int ly = y;
		sData[sy * 18 + 0] = (lx >= 0) ? buffer[ly * w + lx] : 0;
	}

	if (tx == blockDim.x - 1) {
		int rx = x + 1;
		int ly = y;
		sData[sy * 18 + 17] = (rx < w) ? buffer[ly * w + rx] : 0;
	}

	// top/bottom
	if (ty == 0) {
		int ty2 = y - 1;
		int tx2 = x;
		sData[0 * 18 + sx] = (ty2 >= 0) ? buffer[ty2 * w + tx2] : 0;
	}

	if (ty == blockDim.y - 1) {
		int by = y + 1;
		int tx2 = x;
		sData[17 * 18 + sx] = (by < h) ? buffer[by * w + tx2] : 0;
	}

	// corners (important)
	if (tx == 0 && ty == 0) {
		int lx = x - 1, ly = y - 1;
		sData[0 * 18 + 0] = (lx >= 0 && ly >= 0) ? buffer[ly * w + lx] : 0;
	}

	if (tx == blockDim.x - 1 && ty == 0) {
		int rx = x + 1, ly = y - 1;
		sData[0 * 18 + 17] = (rx < w && ly >= 0) ? buffer[ly * w + rx] : 0;
	}

	if (tx == 0 && ty == blockDim.y - 1) {
		int lx = x - 1, by = y + 1;
		sData[17 * 18 + 0] = (lx >= 0 && by < h) ? buffer[by * w + lx] : 0;
	}

	if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
		int rx = x + 1, by = y + 1;
		sData[17 * 18 + 17] = (rx < w && by < h) ? buffer[by * w + rx] : 0;
	}

	__syncthreads();

	// find neighbors
	int n = 0;
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {

			if (dx == 0 && dy == 0) continue;

			int sIdx = (ty + dy + 1) * 18 + (tx + dx + 1);

			if (sData[sIdx]) n++;
		}
	}

	int value = sData[(ty + 1) * 18 + (tx + 1)];

	if (value > 0) { // currently alive
		if (n < 2 || n > 3) value = 0; // die
	}
	else { // currently dead
		if (n == 3) value = 255; // birth
	}

	output[idx] = value;

	uchar1 pixel = make_uchar1(value);
	surf2Dwrite(pixel, destination, x * sizeof(uchar1), y);
}

__global__ void generateRandomGrid(unsigned char* destination, cudaSurfaceObject_t surface, int w, int h, int seed) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int idx = y * w + x;
	int hash = wang_hash(seed ^ x * y);
	int value = (hash & 1) ? 255 : 0;
	destination[idx] = value;
	uchar1 pixel = make_uchar1(value);
	surf2Dwrite(pixel, surface, x * sizeof(uchar1), y);
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
void cudaRegisterTexture(GLuint texture) {
	cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void cudaAllocateBuffer(int w, int h) {
	cudaMalloc(&oldBuffer, w * h * sizeof(unsigned char));
	cudaMalloc(&newBuffer,w*h*sizeof(unsigned char));
}

void cudaFreeBuffer() {
	cudaFree(&oldBuffer);
	cudaFree(&newBuffer);
}

void cudaFillRandom(int w, int h, int seed) {
	cudaSurfaceObject_t surface = mapCudaResource();

	dim3 block(16, 16);
	dim3 grid((w + 15) / 16, (h + 15) / 16);
	generateRandomGrid << <grid, block >> > (oldBuffer, surface, w, h,seed);

	unmapCudaResource(surface);
}

void cudaUpdateTexture(int w, int h) {
	cudaSurfaceObject_t surface = mapCudaResource();

	dim3 block(16, 16);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	updateGrid << <grid, block >> > (oldBuffer, newBuffer, surface, w, h);
	
	unmapCudaResource(surface);

	// swap buffers
	unsigned char* tmp = oldBuffer;
	oldBuffer = newBuffer;
	newBuffer = tmp;
}
