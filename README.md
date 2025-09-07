# CUDA Conway's Game of Life

Conway's Game of Life, but GPU-accelerated using CUDA. Rendering with OpenGL.

Launches full screen in Full HD (1920x1080) resolution with random seed.

## Facts

- Average kernel execution time (including copy from device buffer to shared GL texture also on device and render) on GTX1650:
    - 4K (4096x2160) is around 16ms and it renders with ~60 FPS.
    - Full HD (1920x1080) is around 11ms and it renders with ~60+ FPS.

- Much faster compared to CPU calculation and same GL rendering.

Video of app in work (1920x1080): [link](https://nulldog.xyz/images/cudaconway.mp4)