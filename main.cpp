#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>

#include <GL/glew.h>
#include <glfw3.h>
GLFWwindow* window;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
using namespace glm;

#include "common/shader.hpp"

#define WIDTH  4096
#define HEIGHT 2160

std::string VertexShader = R"(
#version 330 core
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inTex;
out vec2 TexCoord;

void main() {
    gl_Position = vec4(inPos, 0.0, 1.0);
    TexCoord = inTex;
}
)";

std::string FragmentShader = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;

void main() {
    float v = texture(tex, TexCoord).r; // red channel
    FragColor = vec4(v, v, v, 1.0);     // grayscale
}

)";

GLfloat quadVertices[] = {
	// pos      // tex
	-1.0f, -1.0f, 0.0f, 0.0f,
	 1.0f, -1.0f, 1.0f, 0.0f,
	 1.0f,  1.0f, 1.0f, 1.0f,

	-1.0f, -1.0f, 0.0f, 0.0f,
	 1.0f,  1.0f, 1.0f, 1.0f,
	-1.0f,  1.0f, 0.0f, 1.0f
};

int main() {
	HMODULE hDll;
	typedef void(*CudaRegisterTexture)(GLuint texture);
	CudaRegisterTexture cudaRegisterTexture;
	typedef void (*CudaUpdateTexture)(int w, int h);
	CudaUpdateTexture cudaUpdateTexture;
	typedef void (*CudaFillRandom)(int w, int h, int seed);
	CudaFillRandom cudaFillRandom;
	typedef void(*CudaAllocateBuffer)(int w, int h);
	CudaAllocateBuffer cudaAllocateBuffer;
	typedef void(*CudaFreeBuffer)();
	CudaFreeBuffer cudaFreeBuffer;

	hDll = LoadLibraryA("kernel.dll");
	if (!hDll) {
		MessageBoxA(NULL, "Failed to load kernel.dll", "Fatal error", MB_OK | MB_ICONERROR);
		return -1;
	}
	cudaRegisterTexture = (CudaRegisterTexture)GetProcAddress(hDll, "cudaRegisterTexture");
	cudaUpdateTexture = (CudaUpdateTexture)GetProcAddress(hDll, "cudaUpdateTexture");
	cudaFillRandom = (CudaFillRandom)GetProcAddress(hDll, "cudaFillRandom");
	cudaAllocateBuffer = (CudaAllocateBuffer)GetProcAddress(hDll, "cudaAllocateBuffer");
	cudaFreeBuffer = (CudaFreeBuffer)GetProcAddress(hDll, "cudaFreeBuffer");

	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Get primary monitor
	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	if (!monitor) {
		std::cerr << "Failed to get primary monitor\n";
		return -1;
	}

	// Get monitor resolution
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	// Create fullscreen window
	GLFWwindow* window = glfwCreateWindow(
		mode->width, mode->height, "OpenGL Fullscreen", monitor, NULL
	);

	if (!window) {
		std::cerr << "Failed to create GLFW window\n";
		glfwTerminate();
		return -1;
	}
	if (!window) return -1;
	glfwMakeContextCurrent(window);

	glewExperimental = true;
	if (glewInit() != GLEW_OK) return -1;

	// init texture
	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, WIDTH, HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	cudaAllocateBuffer(WIDTH, HEIGHT);
	cudaRegisterTexture(textureID);
	int seed = rand() % 9999999999999;
	cudaFillRandom(WIDTH, HEIGHT, seed);

	// init vao & vbo
	GLuint vao, vbo;
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));

	glBindTexture(GL_TEXTURE_2D, 0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	glfwSwapInterval(1);
	GLuint shaderProgram = LoadShaders(VertexShader, FragmentShader);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		cudaUpdateTexture(WIDTH, HEIGHT);

		glUseProgram(shaderProgram);
		glBindVertexArray(vao);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
		glBindVertexArray(0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	cudaFreeBuffer();
	glfwTerminate();
	return 0;
}