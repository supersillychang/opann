//
//  oclworkengine.cpp
//  oclengine
//
//  Created by Chang Sun on 12/19/17.
//  Copyright © 2017 Chang Sun. All rights reserved.
//

#include "oclworkengine.h"

void OclWorkEngine::debugOut(const char* errMsg)
{
    if (useDebug) {
        std::cerr << errMsg << std::endl;
    }
}

OclWorkEngine::OclWorkEngine(int index, bool enableProfiling)
{
    this->enableProfiling = enableProfiling;
    
    // Create an OpenCL context on first available platform
    createContext();
    if (context == NULL) {
        debugOut("Failed to create OpenCL context.");
    }
    
    // Create a command-queue on the first device available on the created context
    createCommandQueue(index);
    if (commandQueue == NULL) {
        debugOut("Failed to create command queue.");
    }
}

OclWorkEngine::~OclWorkEngine()
{
    cleanUp();
}

//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
void OclWorkEngine::createContext()
{
    if (context != NULL) {
        clReleaseContext(context);
    }
    
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    
    // Select the first available OpenCL platform.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        debugOut("Failed to find any OpenCL platforms.");
        return;
    }
    
    // Create OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        debugOut("Could not create GPU context, trying CPU...");
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS) {
            debugOut("Failed to create an OpenCL GPU or CPU context.");
            return;
        }
    }
}

//  Create a command queue on the first device available on the context
void OclWorkEngine::createCommandQueue(int index)
{
    if (commandQueue != NULL) {
        clReleaseCommandQueue(commandQueue);
    }
    
    cl_int errNum;
    cl_device_id *devices;
    size_t deviceBufferSize = -1;
    
    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        debugOut("Failed call to clGetContextInfo(...,CL_CONTEXT_DEVICES,...)");
        return;
    }
    
    if (deviceBufferSize <= 0) {
        debugOut("No devices available.");
        return;
    }
    
    // Allocate memory for the devices buffer
    size_t deviceCount = deviceBufferSize / sizeof(cl_device_id);
    devices = new cl_device_id[deviceCount];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS) {
        delete [] devices;
        debugOut("Failed to get device IDs");
        return;
    }
    
    int deviceNum = (int)deviceCount - 1 - (index % deviceCount);
    
    // Use last GPU device: normally this would be the dedicated GPU (d-GPU)
    cl_command_queue_properties properties = enableProfiling ? CL_QUEUE_PROFILING_ENABLE : 0;
    commandQueue = clCreateCommandQueue(context, devices[deviceNum], properties, NULL);
    if (commandQueue == NULL) {
        delete [] devices;
        debugOut("Failed to create commandQueue for device 0");
        return;
    }
    
    device = devices[deviceNum];
    delete [] devices;
    
    /*
    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    std::cout << "Max clock frequency (Hz): " << clock_frequency << std::endl;
    
    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    std::cout << "Device max workgroup size: " << max_work_group_size << std::endl;
    
    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    cl_uint max_compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    std::cout << "Max compute units: " << max_compute_units << std::endl;
     */
}

//  Create an OpenCL program from the kernel source file
void OclWorkEngine::createProgram(const char* fileName)
{
    if (program != NULL) {
        releaseKernels();
        clReleaseProgram(program);
    }
    
    cl_int errNum;
    
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        debugOut("Failed to open file for reading: ");
        debugOut(fileName);
        return;
    }
    
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL) {
        debugOut("Failed to create CL program from source.");
        return;
    }
    
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        debugOut("Error in kernel: ");
        debugOut(buildLog);
        clReleaseProgram(program);
        return;
    }
}

// Create an OpenCL kernel given the kernel name
void OclWorkEngine::createKernel(const char* kernelName)
{
    if (kernels.find(kernelName) != kernels.end()) {
        debugOut("Kernel created already, nothing to do.");
        return;
    }
    
    cl_kernel kernel = clCreateKernel(program, kernelName, NULL);
    if (kernel == NULL) {
        debugOut("Failed to create kernel");
        return;
    }
    
    kernels[kernelName] = kernel;
    memObjects[kernel] = std::unordered_map<int, cl_mem>();
    
    /*
    size_t max_work_group_size;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    std::cout << kernelName << " kernel max workgroup size: " << max_work_group_size << std::endl;
     */
}

//  Set kernel arguments, create memory objects used as the arguments to the kernel if necessary
cl_ulong OclWorkEngine::setKernelArg(const char *kernelName, int argIndex, cl_mem_flags memFlag, size_t size, const void *argValue,
                                     cl_bool blocking)
{
    if (kernels.find(kernelName) == kernels.end()) {
        debugOut("Failed to set kernel argument: kernel name not found.");
        return 0;
    }
    
    cl_kernel kernel = kernels[kernelName];
    if (!memFlag) {
        cl_int errNum = clSetKernelArg(kernel, argIndex, size, argValue);
        if (errNum != CL_SUCCESS) {
            debugOut("Error setting kernel arguments.");
        }
        return 0;
    }
    
    // Release previously allocated memory object (TODO: optimize)
    if (memObjects[kernel].find(argIndex) != memObjects[kernel].end()) {
        clReleaseMemObject(memObjects[kernel][argIndex]);
    }
    
    // Create new memory object and add to map
    memObjects[kernel][argIndex] = clCreateBuffer(context, memFlag, size, NULL, NULL);
    if (memObjects[kernel][argIndex] == NULL) {
        debugOut("Error creating memory objects.");
        return 0;
    }
    
    cl_event event;
    cl_event* event_ptr = enableProfiling ? &event : NULL;
    cl_int errNum = clEnqueueWriteBuffer(commandQueue, memObjects[kernel][argIndex], blocking, 0, size, argValue, 0, NULL, event_ptr);
    if (errNum != CL_SUCCESS) {
        debugOut("Failed to enqueue buffer write.");
        return 0;
    }
    
    errNum = clSetKernelArg(kernel, argIndex, sizeof(cl_mem), &memObjects[kernel][argIndex]);
    if (errNum != CL_SUCCESS) {
        debugOut("Error setting kernel arguments.");
        clReleaseMemObject(memObjects[kernel][argIndex]);
        memObjects[kernel].erase(argIndex);
    }
    
    return getEventTime(event_ptr);
}

// Enqueue NDRange command
cl_ulong OclWorkEngine::enqueueKernel(const char *kernelName, cl_uint workDim, const size_t *globalWorkSize, const size_t *localWorkSize,
                                      bool blocking)
{
    if (kernels.find(kernelName) == kernels.end()) {
        debugOut("Failed to enqueue kernel: kernel name not found.");
        return 0;
    }
    
    cl_kernel kernel = kernels[kernelName];
    cl_event event;
    cl_event* event_ptr = enableProfiling ? &event : NULL;
    cl_int errNum = clEnqueueNDRangeKernel(commandQueue, kernel, workDim, NULL, globalWorkSize, localWorkSize, 0, NULL, event_ptr);
    if (errNum != CL_SUCCESS) {
        debugOut("Error queuing kernel for execution.");
    }
    
    if (blocking) {
        clFinish(commandQueue);
    }
    
    return getEventTime(event_ptr);
}

// Copy back memory from GPU
cl_ulong OclWorkEngine::enqueueReadBuffer(const char *kernelName, int argIndex, size_t size, void *result, cl_bool blocking)
{
    if (kernels.find(kernelName) == kernels.end()) {
        debugOut("Failed to enqueue buffer read: kernel name not found.");
        return 0;
    }
    
    cl_kernel kernel = kernels[kernelName];
    if (memObjects[kernel].find(argIndex) == memObjects[kernel].end()) {
        debugOut("Failed to enqueue buffer read: invalid argument index.");
        return 0;
    }
    
    cl_event event;
    cl_event* event_ptr = enableProfiling ? &event : NULL;
    cl_int errNum = clEnqueueReadBuffer(commandQueue, memObjects[kernel][argIndex], blocking, 0, size, result, 0, NULL, event_ptr);
    if (errNum != CL_SUCCESS) {
        debugOut("Error reading result buffer.");
        return 0;
    }
    
    return getEventTime(event_ptr);
}

// Release OpenCL kernels, clean up kernel map
void OclWorkEngine::releaseKernels()
{
    releaseMemObjects();
    for (auto i = kernels.begin(); i != kernels.end(); ++i) {
        clReleaseKernel(i->second);
    }
    kernels.clear();
}

// Release OpenCL memory objects, clean up memory object map
void OclWorkEngine::releaseMemObjects()
{
    for (auto i = memObjects.begin(); i != memObjects.end(); ++i) {
        for (auto j = i->second.begin(); j != i->second.end(); ++j) {
            clReleaseMemObject(j->second);
        }
    }
    memObjects.clear();
}

//  Cleanup any created OpenCL resources
void OclWorkEngine::cleanUp()
{
    releaseKernels();
    
    if (commandQueue != NULL) {
        clReleaseCommandQueue(commandQueue);
    }
    
    if (program != NULL) {
        clReleaseProgram(program);
    }
    
    if (context != NULL) {
        clReleaseContext(context);
    }
}

cl_ulong OclWorkEngine::getEventTime(cl_event* event)
{
    if (event == NULL) {
        return 0;
    }
    
    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    
    return time_end - time_start;
}
