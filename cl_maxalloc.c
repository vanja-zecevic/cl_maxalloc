/*
 * cl_maxalloc.c
 * 
 * Copyright 2012 Vanja Zecevic
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * -----------------------------------------------------------------------------
 * Description:
 * This program is used to test the maximum amount of usable memory
 * on an OpenCL GPU device. The program allocates the desired amount
 * of memory in chunks and attempts to write to it. Finally, if the
 * number of chunks is 1 or 2, a simple test kernel is run and the
 * results are verified.
 */

#include <stdio.h>
#include <CL/cl.h>

#define BIGINT long unsigned int

/*----------------------------------------------------------------------------*/
/* Test kernel source.  */

const char * krn_src =
"__kernel void test_1chunk( __global int * buff )                            \n"
"{                                                                           \n"
"unsigned int idx;                                                           \n"
"idx = get_group_id(0)*get_local_size(0) + get_local_id(0);                  \n"
"buff[idx] += 1;                                                             \n"
"}                                                                           \n"
"__kernel void test_2chunk( __global int * buff_0,                           \n"
"  __global int * buff_1)                                                    \n"
"{                                                                           \n"
"unsigned int idx;                                                           \n"
"idx = get_group_id(0)*get_local_size(0) + get_local_id(0);                  \n"
"buff_0[idx] += 1;                                                           \n"
"buff_1[idx] += 1;                                                           \n"
"}                                                                           \n"
;
/*----------------------------------------------------------------------------*/
int main (int argc, char * argv[])
{
BIGINT chunk = 10;
BIGINT maxmem = 2000;
BIGINT iChunk;
BIGINT nChunk;
BIGINT iX;
BIGINT nX;
BIGINT nAccessed;
int iArg;
int iT;
size_t glo;
size_t loc;

cl_platform_id platform;
cl_device_id device;
cl_context Context;
cl_command_queue CmdQueue;
cl_int err_tr = CL_SUCCESS;
cl_ulong global_size;
cl_kernel test_1chunk;
cl_kernel test_2chunk;
cl_mem * buffers_dev;
cl_program prog;
int ** buffers_host;
size_t cl_ret_size;
char * cl_log;

/*----------------------------------------------------------------------------*/
/* Get flags.  */
if (argc <=6 ) {
    for (iArg=1; iArg<argc; iArg++) {
        if (!strcmp(argv[iArg],"--help")) {
            printf(
  "\nUSAGE:\n"
  "cl_maxalloc <flags>\n"
  " This program is used to test the maximum amount of usable memory\n"
  " on an OpenCL GPU device. The program allocates the desired amount\n"
  " of memory in chunks and attempts to write to it. Finally, if the\n"
  " number of chunks is 1 or 2, a simple test kernel is run and the\n"
  " results are verified.\n"
  "\n"
  "EXAMPLE:\n"
  "cl_maxalloc --chunk 10 --maxmem 2000\n"
  "\n"
  "FLAGS:\n"
  "--help   Prints this message\n"
  "--chunk  The size of each chunk to be allocated in MB (default 10 MB)\n"
  "--maxmem The maximum memory to allocate in MB (default 2000 MB)\n"
              );
            exit(1);
        }
        else if (!strcmp(argv[iArg],"--chunk"))  chunk  = atoi(argv[iArg+1]); 
        else if (!strcmp(argv[iArg],"--maxmem")) maxmem = atoi(argv[iArg+1]);
    }
}

nChunk = maxmem/chunk;
nX = (chunk*(BIGINT)1e6)/sizeof(int);

/*----------------------------------------------------------------------------*/
/* Initialize OpenCL devices.  */
err_tr = clGetPlatformIDs(1, &platform, NULL);
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
Context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
CmdQueue = clCreateCommandQueue(Context, device, 0, NULL);

clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
  &global_size, NULL);
printf("Created OpenCL context.\n"
       "Global memory size: %li MB\n",
        global_size/(BIGINT)1e6);

/*----------------------------------------------------------------------------*/
/* First allocate buffers.  */
buffers_dev = (cl_mem*)malloc(nChunk*sizeof(cl_mem));
for (iChunk=0; iChunk<nChunk; iChunk++) {
    *(buffers_dev+iChunk) = clCreateBuffer(Context, CL_MEM_READ_WRITE,
      nX*sizeof(int), NULL, &err_tr);
    if (err_tr != CL_SUCCESS) {
        printf("\033[1;31mFAIL\033[00m, error %i\n", err_tr);
        exit(1);
    }
}

printf("Allocated %lu MB\n", iChunk*nX*sizeof(int)/(BIGINT)1e6);
nChunk = iChunk;

/*----------------------------------------------------------------------------*/
/* Now try to initialize buffers.  */
printf("nX %lu\n", nX);
/* Create an array of initialized buffers.  */
buffers_host = (int**)malloc(nChunk*sizeof(int*));
for (iChunk=0; iChunk<nChunk; iChunk++) {
    *(buffers_host+iChunk) = (int*)malloc(nX*sizeof(int));
    for (iX=0; iX<nX; iX++) *(*(buffers_host+iChunk)+iX) = 0;
}

for (iChunk=0; iChunk<nChunk; iChunk++) {
    err_tr = clEnqueueWriteBuffer(CmdQueue, *(buffers_dev+iChunk), CL_TRUE, 0,
      nX*sizeof(int), *(buffers_host+iChunk), 0, NULL, NULL);
    if (err_tr != CL_SUCCESS) {
        printf("\033[1;31mFAIL\033[00m, error %i\n", err_tr);
        exit(1);
    }
}

printf("Initialized %lu MB\n", iChunk*nX*sizeof(int)/(BIGINT)1e6);
nAccessed = iChunk;

/*----------------------------------------------------------------------------*/
/*  Build the simple test kernel.  */
prog = clCreateProgramWithSource(Context, 1,
  &krn_src, 0, &err_tr);
if (err_tr != CL_SUCCESS) {
    printf("clCreateProgramWithSource %i\n", err_tr);
    exit(1);
}

err_tr = clBuildProgram(prog, 0, 0, "-cl-opt-disable", 0, 0);

if (err_tr != CL_SUCCESS) {
    /* Print errors if unsuccessful.  */
    err_tr = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
      &cl_ret_size);
    cl_log = (char*)malloc(cl_ret_size+1);
    err_tr = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
      cl_ret_size, cl_log, NULL);

    *(cl_log+cl_ret_size) = '\0';

    /* Print any messages.  */
    printf("%s\n", cl_log);
    free(cl_log);
    exit(1);
}
test_1chunk = clCreateKernel(prog, "test_1chunk", NULL );
test_2chunk = clCreateKernel(prog, "test_2chunk", NULL );

/*----------------------------------------------------------------------------*/
/* Run the test kernel.  */
loc = 64;
glo = nX;
if (glo%loc!=0) {
    printf("Bad nX, needs to be a multiple of 64\n");
    exit(1);
}

if (nChunk==1) {
    clSetKernelArg(test_1chunk, 0, sizeof(cl_mem), (buffers_dev+0));
    for (iT=0; iT<100; iT++) {
        err_tr = clEnqueueNDRangeKernel(CmdQueue, test_1chunk, 1, NULL,
          &glo, &loc, 0, 0, 0);
        if (err_tr != CL_SUCCESS) {
            printf("\033[1;31mFAIL\033[00m, error %i\n", err_tr);
            exit(1);
        }
    }
} else if (nChunk==2) {
    clSetKernelArg(test_2chunk, 0, sizeof(cl_mem), (buffers_dev+0));
    clSetKernelArg(test_2chunk, 1, sizeof(cl_mem), (buffers_dev+1));
    for (iT=0; iT<100; iT++) {
        err_tr = clEnqueueNDRangeKernel(CmdQueue, test_2chunk, 1, NULL,
          &glo, &loc, 0, 0, 0);
        if (err_tr != CL_SUCCESS) {
            printf("\033[1;31mFAIL\033[00m, error %i\n", err_tr);
            exit(1);
        }
    }
} else {
    printf("Unable to run test kernel, try less chunks.\n");
    exit(1);
}

clFinish(CmdQueue);
printf("test kernel completed.\n");

/*----------------------------------------------------------------------------*/
/* Now check buffers.  */

for (iChunk=0; iChunk<nChunk; iChunk++) {
    err_tr = clEnqueueReadBuffer(CmdQueue, *(buffers_dev+iChunk), CL_TRUE, 0,
      nX*sizeof(int), *(buffers_host+iChunk), 0, NULL, NULL);
    if (err_tr != CL_SUCCESS) {
        printf("\033[1;31mFAIL\033[00m, error %i\n", err_tr);
        exit(1);
    }
}
for (iChunk=0; iChunk<nChunk; iChunk++)
  for (iX=0; iX<nX; iX++) if (*(*(buffers_host+iChunk)+iX)!=100) {
    printf("\033[1;31mFAIL\033[00m\n");
    exit(1);
}

printf("Checked %lu MB\n", iChunk*nX*sizeof(int)/(BIGINT)1e6);

/*----------------------------------------------------------------------------*/
/* Clean up host and device buffers.  */
for (iChunk=0; iChunk<nAccessed; iChunk++)
  free(*(buffers_host+iChunk));
for (iChunk=0; iChunk<nChunk; iChunk++)
  clReleaseMemObject(*(buffers_dev+iChunk));
free(buffers_host);
free(buffers_dev);
clReleaseCommandQueue(CmdQueue);
clReleaseContext(Context);

return 0;
}
/*----------------------------------------------------------------------------*/

