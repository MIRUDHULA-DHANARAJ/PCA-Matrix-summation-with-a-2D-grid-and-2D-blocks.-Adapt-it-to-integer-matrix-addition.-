# PCA-Matrix-summation-with-a-2D-grid-and-2D-blocks.-Adapt-it-to-integer-matrix-addition.-

## Aim: 
To perform PCA matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## Procedure:
# STEP 1:
Allocate memory on the device for the input and output arrays.
# STEP 2:
Copy the input array from the host to the device.
# STEP 3:
Set the grid and block dimensions for the kernel.
# STEP 4:
Launch the kernel with the grid and block dimensions and the input and output arrays.
# STEP 5:
Copy the output array from the device to the host.
# STEP 6:
Free the memory on the device.

## Program:

```
Developed by: Mirudhula D

Register Number: 212221230060

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>



void initialData(int *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (int)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(int *A, int *B, int *C, const int nx,
                     const int ny)
{
    int *ia = A;
    int *ib = B;
    int *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}


void checkResult(int *hostRef, int *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// grid 2D block 2D
_global_ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 3;
    int ny = 1 << 3;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    int *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    // initialize data at host side
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 3;
    int dimy = 3;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
           grid.y,
           block.x, block.y, iElaps);
    // check kernel error
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
```

## Output:

```
root@MidPC:/home/student/Desktop# nvcc first.cu
root@MidPC:/home/student/Desktop# ./a.out
./a.out Starting...
Using Device 0: NVIDIA GeForce GTX 1660 SUPER
Matrix size: nx 16384 ny 16384
Matrix initialization elapsed 6.199756 sec
sumMatrixOnHost elapsed 0.880898 sec
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 0.012740 sec
Arrays match.
```
![2](https://github.com/MIRUDHULA-DHANARAJ/PCA-Matrix-summation-with-a-2D-grid-and-2D-blocks.-Adapt-it-to-integer-matrix-addition.-/assets/94828147/736658c1-9fda-4675-80fc-4d03394e2999)



## Result:
Thus the program to perform PCA matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition has been successfully executed.
