#define GLFW_INCLUDE_NONE 
#include "glad.h"
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda_gl_interop.h>

#include <iostream>

void cudaRegisterTexture(GLuint texture);
void cudaAllocateBuffer(int w, int h);
void cudaFreeBuffer();
void cudaFillRandom(int w, int h, int seed);
void cudaUpdateTexture(int w, int h);